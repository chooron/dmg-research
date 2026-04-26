from typing import Any, Optional, Union

import torch

from hydrodl2.core.calc import change_param_range, uh_conv, uh_gamma


class Hbv_2f(torch.nn.Module):
    """HBV 2.0 fast training variant with fixed dynamic parameters.

    This model is intentionally specialized:
    - only `streamflow` is supported
    - only `parBETA`, `parK0`, `parBETAET` are dynamic
    - all remaining physical parameters are static
    - warmup remains in the computation graph, but is trimmed from outputs
    """

    DEFAULT_OUTPUT_KEYS = ("streamflow",)
    DYNAMIC_PARAM_NAMES = ("parBETA", "parK0", "parBETAET")
    STATIC_PARAM_NAMES = (
        "parFC",
        "parK1",
        "parK2",
        "parLP",
        "parPERC",
        "parUZL",
        "parTT",
        "parCFMAX",
        "parCFR",
        "parCWH",
        "parC",
        "parRT",
        "parAC",
    )
    ROUTING_LEN = 15

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.name = "HBV 2.0UH fast"
        self.config = config
        self.initialize = False
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.variables = ["prcp", "tmean", "pet"]
        self.routing = True
        self.comprout = False
        self.nearzero = 1e-5
        self.nmul = 1
        self.output_keys = self.DEFAULT_OUTPUT_KEYS
        self.device = device
        self.parameter_bounds = {
            "parBETA": [1.0, 6.0],
            "parFC": [50, 1000],
            "parK0": [0.05, 0.9],
            "parK1": [0.01, 0.5],
            "parK2": [0.001, 0.2],
            "parLP": [0.2, 1],
            "parPERC": [0, 10],
            "parUZL": [0, 100],
            "parTT": [-2.5, 2.5],
            "parCFMAX": [0.5, 10],
            "parCFR": [0, 0.1],
            "parCWH": [0, 0.2],
            "parBETAET": [0.3, 5],
            "parC": [0, 1],
            "parRT": [0, 20],
            "parAC": [0, 2500],
        }
        self.routing_parameter_bounds = {
            "rout_a": [0, 2.9],
            "rout_b": [0, 6.5],
        }

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

        if config is not None:
            self.warm_up = config.get("warm_up", self.warm_up)
            self.warm_up_states = config.get(
                "warm_up_states",
                self.warm_up_states,
            )
            self.variables = config.get("variables", self.variables)
            self.routing = config.get("routing", self.routing)
            self.comprout = config.get("comprout", self.comprout)
            self.nearzero = config.get("nearzero", self.nearzero)
            self.nmul = config.get("nmul", self.nmul)
            output_keys = config.get("output_keys", self.output_keys)
            if isinstance(output_keys, str):
                output_keys = [output_keys]
            self.output_keys = tuple(dict.fromkeys(output_keys))

        self._validate_output_keys()
        self._compiled_hbv_step = self._build_compiled_hbv_step()
        self.set_parameters()

    def _validate_output_keys(self) -> None:
        if set(self.output_keys) != {"streamflow"}:
            raise ValueError(
                "Hbv_2f only supports output_keys=['streamflow'] in fast mode."
            )

    def _build_compiled_hbv_step(self):
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                "Hbv_2f requires torch.compile, but this PyTorch build does not provide it."
            )

        try:
            return torch.compile(self._hbv_step_impl, fullgraph=True)
        except Exception as exc:
            raise RuntimeError(
                "Hbv_2f failed to build the torch.compile step function."
            ) from exc

    def set_parameters(self) -> None:
        self.phy_param_names = tuple(self.parameter_bounds.keys())
        self.static_param_names = self.STATIC_PARAM_NAMES
        if self.routing:
            self.routing_param_names = tuple(self.routing_parameter_bounds.keys())
        else:
            self.routing_param_names = ()

        self.learnable_param_count1 = len(self.DYNAMIC_PARAM_NAMES) * self.nmul
        self.learnable_param_count2 = (
            len(self.static_param_names) * self.nmul
            + len(self.routing_param_names)
        )
        self.learnable_param_count = (
            self.learnable_param_count1 + self.learnable_param_count2
        )

    @staticmethod
    def _hbv_step_impl(
        PRECIP: torch.Tensor,
        TEMP: torch.Tensor,
        PET: torch.Tensor,
        SNOWPACK: torch.Tensor,
        MELTWATER: torch.Tensor,
        SM: torch.Tensor,
        SUZ: torch.Tensor,
        SLZ: torch.Tensor,
        parBETA: torch.Tensor,
        parFC: torch.Tensor,
        parK0: torch.Tensor,
        parK1: torch.Tensor,
        parK2: torch.Tensor,
        parLP: torch.Tensor,
        parPERC: torch.Tensor,
        parUZL: torch.Tensor,
        parTT: torch.Tensor,
        parCFMAX: torch.Tensor,
        parCFR: torch.Tensor,
        parCWH: torch.Tensor,
        parBETAET: torch.Tensor,
        parC: torch.Tensor,
        nearzero: float,
    ) -> tuple[torch.Tensor, ...]:
        RAIN = PRECIP * (TEMP >= parTT).to(PRECIP.dtype)
        SNOW = PRECIP * (TEMP < parTT).to(PRECIP.dtype)

        SNOWPACK = SNOWPACK + SNOW
        melt = torch.clamp(parCFMAX * (TEMP - parTT), min=0.0)
        melt = torch.min(melt, SNOWPACK)
        MELTWATER = MELTWATER + melt
        SNOWPACK = SNOWPACK - melt

        refreezing = torch.clamp(parCFR * parCFMAX * (parTT - TEMP), min=0.0)
        refreezing = torch.min(refreezing, MELTWATER)
        SNOWPACK = SNOWPACK + refreezing
        MELTWATER = MELTWATER - refreezing

        tosoil = torch.clamp(MELTWATER - (parCWH * SNOWPACK), min=0.0)
        MELTWATER = MELTWATER - tosoil

        soil_wetness = torch.clamp((SM / parFC) ** parBETA, min=0.0, max=1.0)
        recharge = (RAIN + tosoil) * soil_wetness
        SM = SM + RAIN + tosoil - recharge

        excess = torch.clamp(SM - parFC, min=0.0)
        SM = SM - excess

        evapfactor = torch.clamp(
            (SM / (parLP * parFC)) ** parBETAET,
            min=0.0,
            max=1.0,
        )
        ETact = torch.min(SM, PET * evapfactor)
        SM = torch.clamp(SM - ETact, min=nearzero)

        capillary = torch.min(
            SLZ,
            parC * SLZ * (1.0 - torch.clamp(SM / parFC, max=1.0)),
        )
        SM = torch.clamp(SM + capillary, min=nearzero)
        SLZ = torch.clamp(SLZ - capillary, min=nearzero)

        SUZ = SUZ + recharge + excess
        PERC = torch.min(SUZ, parPERC)
        SUZ = SUZ - PERC

        Q0 = parK0 * torch.clamp(SUZ - parUZL, min=0.0)
        SUZ = SUZ - Q0
        Q1 = parK1 * SUZ
        SUZ = SUZ - Q1

        SLZ = torch.clamp(SLZ + PERC, min=0.0)
        Q2 = parK2 * SLZ
        SLZ = SLZ - Q2

        return (
            SNOWPACK,
            MELTWATER,
            SM,
            SUZ,
            SLZ,
            Q0 + Q1 + Q2,
        )

    def _initialize_states(self, n_grid: int) -> tuple[torch.Tensor, ...]:
        state = torch.zeros(
            [n_grid, self.nmul],
            dtype=torch.float32,
            device=self.device,
        ) + 0.001
        return (
            state.clone(),
            state.clone(),
            state.clone(),
            state.clone(),
            state.clone(),
        )

    def unpack_parameters(
        self,
        parameters: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        static_param_count = len(self.static_param_names)
        phy_dy_params = parameters[0].view(
            parameters[0].shape[0],
            parameters[0].shape[1],
            len(self.DYNAMIC_PARAM_NAMES),
            self.nmul,
        )
        phy_static_params = parameters[1][:, : static_param_count * self.nmul].view(
            parameters[1].shape[0],
            static_param_count,
            self.nmul,
        )

        routing_params = None
        if self.routing:
            routing_params = parameters[1][:, static_param_count * self.nmul :]

        return phy_dy_params, phy_static_params, routing_params

    def descale_dynamic_parameters(
        self,
        phy_dy_params: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        parBETA = change_param_range(
            phy_dy_params[:, :, 0, :],
            self.parameter_bounds["parBETA"],
        )
        parK0 = change_param_range(
            phy_dy_params[:, :, 1, :],
            self.parameter_bounds["parK0"],
        )
        parBETAET = change_param_range(
            phy_dy_params[:, :, 2, :],
            self.parameter_bounds["parBETAET"],
        )
        return parBETA, parK0, parBETAET

    def descale_static_parameters(
        self,
        phy_static_params: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        parFC = change_param_range(
            phy_static_params[:, 0, :],
            self.parameter_bounds["parFC"],
        )
        parK1 = change_param_range(
            phy_static_params[:, 1, :],
            self.parameter_bounds["parK1"],
        )
        parK2 = change_param_range(
            phy_static_params[:, 2, :],
            self.parameter_bounds["parK2"],
        )
        parLP = change_param_range(
            phy_static_params[:, 3, :],
            self.parameter_bounds["parLP"],
        )
        parPERC = change_param_range(
            phy_static_params[:, 4, :],
            self.parameter_bounds["parPERC"],
        )
        parUZL = change_param_range(
            phy_static_params[:, 5, :],
            self.parameter_bounds["parUZL"],
        )
        parTT = change_param_range(
            phy_static_params[:, 6, :],
            self.parameter_bounds["parTT"],
        )
        parCFMAX = change_param_range(
            phy_static_params[:, 7, :],
            self.parameter_bounds["parCFMAX"],
        )
        parCFR = change_param_range(
            phy_static_params[:, 8, :],
            self.parameter_bounds["parCFR"],
        )
        parCWH = change_param_range(
            phy_static_params[:, 9, :],
            self.parameter_bounds["parCWH"],
        )
        parC = change_param_range(
            phy_static_params[:, 10, :],
            self.parameter_bounds["parC"],
        )

        return (
            parFC,
            parK1,
            parK2,
            parLP,
            parPERC,
            parUZL,
            parTT,
            parCFMAX,
            parCFR,
            parCWH,
            parC,
        )

    def descale_rout_parameters(
        self,
        routing_params: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "rout_a": change_param_range(
                routing_params[:, 0],
                self.routing_parameter_bounds["rout_a"],
            ),
            "rout_b": change_param_range(
                routing_params[:, 1],
                self.routing_parameter_bounds["rout_b"],
            ),
        }

    def _simulate_runoff(
        self,
        forcing: torch.Tensor,
        states: tuple[torch.Tensor, ...],
        parBETA: torch.Tensor,
        parK0: torch.Tensor,
        parBETAET: torch.Tensor,
        static_params: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        n_steps = forcing.size(0)
        n_grid = forcing.size(1)
        if n_steps == 0:
            empty = forcing.new_empty((0, n_grid, self.nmul))
            return empty, states

        (
            parFC,
            parK1,
            parK2,
            parLP,
            parPERC,
            parUZL,
            parTT,
            parCFMAX,
            parCFR,
            parCWH,
            parC,
        ) = static_params

        SNOWPACK, MELTWATER, SM, SUZ, SLZ = states
        P = forcing[:, :, self.variables.index("prcp")]
        T = forcing[:, :, self.variables.index("tmean")]
        PET = forcing[:, :, self.variables.index("pet")]

        Pm = P.unsqueeze(2).repeat(1, 1, self.nmul)
        Tm = T.unsqueeze(2).repeat(1, 1, self.nmul)
        PETm = PET.unsqueeze(2).repeat(1, 1, self.nmul)
        runoff = torch.empty(Pm.size(), dtype=torch.float32, device=self.device)

        for t in range(n_steps):
            try:
                (
                    SNOWPACK,
                    MELTWATER,
                    SM,
                    SUZ,
                    SLZ,
                    Qsim_step,
                ) = self._compiled_hbv_step(
                    Pm[t, :, :],
                    Tm[t, :, :],
                    PETm[t, :, :],
                    SNOWPACK,
                    MELTWATER,
                    SM,
                    SUZ,
                    SLZ,
                    parBETA[t, :, :],
                    parFC,
                    parK0[t, :, :],
                    parK1,
                    parK2,
                    parLP,
                    parPERC,
                    parUZL,
                    parTT,
                    parCFMAX,
                    parCFR,
                    parCWH,
                    parBETAET[t, :, :],
                    parC,
                    self.nearzero,
                )
            except Exception as exc:
                raise RuntimeError(
                    "Hbv_2f recurrent step torch.compile execution failed."
                ) from exc

            runoff[t, :, :] = Qsim_step

        return runoff, (SNOWPACK, MELTWATER, SM, SUZ, SLZ)

    def _aggregate_member_runoff(self, runoff: torch.Tensor) -> torch.Tensor:
        if self.muwts is None:
            return runoff.mean(-1)
        return (runoff * self.muwts).sum(-1)

    def _route_streamflow(self, runoff: torch.Tensor) -> torch.Tensor:
        n_steps = runoff.size(0)
        kernel_steps = max(1, min(n_steps, self.ROUTING_LEN))
        UH = uh_gamma(
            self.routing_param_dict["rout_a"].repeat(kernel_steps, 1).unsqueeze(-1),
            self.routing_param_dict["rout_b"].repeat(kernel_steps, 1).unsqueeze(-1),
            lenF=self.ROUTING_LEN,
        ).permute([1, 2, 0])

        if self.comprout:
            flat_runoff = runoff.view(n_steps, runoff.size(1) * runoff.size(2))
            routed_uh = UH.repeat_interleave(self.nmul, dim=0)
            rf = flat_runoff.unsqueeze(-1).permute([1, 2, 0])
            routed = uh_conv(rf, routed_uh).permute([2, 0, 1]).view(
                n_steps,
                runoff.size(1),
                runoff.size(2),
            )
            return self._aggregate_member_runoff(routed).unsqueeze(-1)

        avg_runoff = self._aggregate_member_runoff(runoff)
        rf = avg_runoff.unsqueeze(-1).permute([1, 2, 0])
        return uh_conv(rf, UH).permute([2, 0, 1])

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: tuple[torch.Tensor, torch.Tensor],
    ) -> Union[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
        x = x_dict["x_phy"]
        self.muwts = x_dict.get("muwts", None)

        phy_dy_params, phy_static_params, routing_params = self.unpack_parameters(
            parameters
        )
        if self.routing:
            self.routing_param_dict = self.descale_rout_parameters(routing_params)

        states = self._initialize_states(x.size(1))
        self.pred_cutoff = min(self.warm_up, x.size(0))

        parBETA, parK0, parBETAET = self.descale_dynamic_parameters(phy_dy_params)
        static_params = self.descale_static_parameters(phy_static_params)
        runoff, final_states = self._simulate_runoff(
            x,
            states,
            parBETA,
            parK0,
            parBETAET,
            static_params,
        )

        if self.initialize:
            return final_states

        if self.routing:
            streamflow = self._route_streamflow(runoff)
        else:
            streamflow = self._aggregate_member_runoff(runoff).unsqueeze(-1)

        return {"streamflow": streamflow[self.pred_cutoff :]}
