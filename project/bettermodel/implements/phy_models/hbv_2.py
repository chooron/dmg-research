from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from hydrodl2.core.calc import change_param_range, uh_conv, uh_gamma


def _hbv2_step(
        precip: torch.Tensor,
        temp: torch.Tensor,
        pet: torch.Tensor,
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
    RAIN = torch.mul(precip, (temp >= parTT).type(torch.float32))
    SNOW = torch.mul(precip, (temp < parTT).type(torch.float32))

    SNOWPACK = SNOWPACK + SNOW
    melt = parCFMAX * (temp - parTT)
    melt = torch.clamp(melt, min=0.0)
    melt = torch.min(melt, SNOWPACK)
    MELTWATER = MELTWATER + melt
    SNOWPACK = SNOWPACK - melt
    refreezing = parCFR * parCFMAX * (parTT - temp)
    refreezing = torch.clamp(refreezing, min=0.0)
    refreezing = torch.min(refreezing, MELTWATER)
    SNOWPACK = SNOWPACK + refreezing
    MELTWATER = MELTWATER - refreezing
    tosoil = MELTWATER - (parCWH * SNOWPACK)
    tosoil = torch.clamp(tosoil, min=0.0)
    MELTWATER = MELTWATER - tosoil

    soil_wetness = (SM / parFC) ** parBETA
    soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
    recharge = (RAIN + tosoil) * soil_wetness

    SM = SM + RAIN + tosoil - recharge
    excess = SM - parFC
    excess = torch.clamp(excess, min=0.0)
    SM = SM - excess

    evapfactor = (SM / (parLP * parFC)) ** parBETAET
    evapfactor = torch.clamp(evapfactor, min=0.0, max=1.0)
    ETact = pet * evapfactor
    ETact = torch.min(SM, ETact)
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
        Q0 + Q1 + Q2,
        Q0,
        Q1,
        Q2,
        ETact,
        SNOWPACK,
        capillary,
        recharge,
        excess,
        evapfactor,
        tosoil + RAIN,
        soil_wetness,
        PERC,
        SM,
        SNOWPACK,
        MELTWATER,
        SM,
        SUZ,
        SLZ,
    )


class Hbv_2(torch.nn.Module):
    """HBV 2.0 ~.

    Multi-component, multi-scale, differentiable PyTorch HBV model with rainfall
    runoff simulation on unit basins.

    Authors
    -------
    -   Yalan Song
    -   (Original NumPy HBV ver.) Beck et al., 2020 (http://www.gloh2o.org/hbv/).
    -   (HBV-light Version 2) Seibert, 2005
        (https://www.geo.uzh.ch/dam/jcr:c8afa73c-ac90-478e-a8c7-929eed7b1b62/HBV_manual_2005.pdf).

    Publication
    -----------
    -   Yalan Song, Tadd Bindas, Chaopeng Shen, et al. "High-resolution
        national-scale water modeling is enhanced by multiscale differentiable
        physics-informed machine learning." Water Resources Research (2025).
        https://doi.org/10.1029/2024WR038928.

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        Device to run the model on.
    """

    def __init__(
            self,
            config: Optional[dict[str, Any]] = None,
            device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.name = 'HBV 2.0UH'
        self.config = config
        self.initialize = False
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.dynamic_params = []
        self.dy_drop = 0.0
        self.variables = ['prcp', 'tmean', 'pet']
        self.routing = True
        self.comprout = False
        self.nearzero = 1e-5
        self.nmul = 1
        self.device = device
        self.parameter_bounds = {
            'parBETA': [1.0, 6.0],
            'parFC': [50, 1000],
            'parK0': [0.05, 0.9],
            'parK1': [0.01, 0.5],
            'parK2': [0.001, 0.2],
            'parLP': [0.2, 1],
            'parPERC': [0, 10],
            'parUZL': [0, 100],
            'parTT': [-2.5, 2.5],
            'parCFMAX': [0.5, 10],
            'parCFR': [0, 0.1],
            'parCWH': [0, 0.2],
            'parBETAET': [0.3, 5],
            'parC': [0, 1],
            'parRT': [0, 20],
            'parAC': [0, 2500],

        }
        self.routing_parameter_bounds = {
            'rout_a': [0, 2.9],
            'rout_b': [0, 6.5],
        }

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config is not None:
            # Overwrite defaults with config values.
            self.warm_up = config.get('warm_up', self.warm_up)
            self.warm_up_states = config.get('warm_up_states', self.warm_up_states)
            self.dy_drop = config.get('dy_drop', self.dy_drop)
            self.dynamic_params = config['dynamic_params'].get(self.__class__.__name__, self.dynamic_params)
            self.variables = config.get('variables', self.variables)
            self.routing = config.get('routing', self.routing)
            self.comprout = config.get('comprout', self.comprout)
            self.nearzero = config.get('nearzero', self.nearzero)
            self.nmul = config.get('nmul', self.nmul)
        self.set_parameters()
        self._step = torch.compile(_hbv2_step, fullgraph=True)

    def set_parameters(self) -> None:
        """Get physical parameters."""
        self.phy_param_names = self.parameter_bounds.keys()
        if self.routing:
            self.routing_param_names = self.routing_parameter_bounds.keys()
        else:
            self.routing_param_names = []

        self.learnable_param_count1 = len(self.dynamic_params) * self.nmul
        self.learnable_param_count2 = (len(self.phy_param_names) - len(self.dynamic_params)) * self.nmul \
                                      + len(self.routing_param_names)
        self.learnable_param_count = self.learnable_param_count1 + self.learnable_param_count2

    def unpack_parameters(
            self,
            parameters: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract physical model and routing parameters from NN output.

        Parameters
        ----------
        parameters
            Unprocessed, learned parameters from a neural network.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple of physical and routing parameters.
        """
        phy_param_count = len(self.parameter_bounds)
        dy_param_count = len(self.dynamic_params)
        dif_count = phy_param_count - dy_param_count

        # Physical dynamic parameters
        phy_dy_params = parameters[0].view(
            parameters[0].shape[0],
            parameters[0].shape[1],
            dy_param_count,
            self.nmul,
        )

        # Physical static parameters
        phy_static_params = parameters[1][:, :dif_count * self.nmul].view(
            parameters[1].shape[0],
            dif_count,
            self.nmul,
        )

        # Routing parameters
        routing_params = None
        if self.routing:
            routing_params = parameters[1][:, dif_count * self.nmul:]

        return phy_dy_params, phy_static_params, routing_params

    def descale_phy_dy_parameters(
            self,
            phy_dy_params: torch.Tensor,
            dy_list: list,
    ) -> dict:
        """Descale physical parameters.

        Parameters
        ----------
        phy_params
            Normalized physical parameters.
        dy_list
            List of dynamic parameters.

        Returns
        -------
        dict
            Dictionary of descaled physical parameters.
        """
        n_steps = phy_dy_params.size(0)
        n_grid = phy_dy_params.size(1)

        # TODO: Fix; if dynamic parameters are not entered in config as they are
        # in HBV params list, then descaling misamtch will occur.
        param_dict = {}
        pmat = torch.ones([1, n_grid, 1]) * self.dy_drop
        for i, name in enumerate(dy_list):
            staPar = phy_dy_params[-1, :, i, :].unsqueeze(0).repeat([n_steps, 1, 1])

            dynPar = phy_dy_params[:, :, i, :]
            drmask = torch.bernoulli(pmat).detach_().to(self.device)

            comPar = dynPar * (1 - drmask) + staPar * drmask
            param_dict[name] = change_param_range(
                param=comPar,
                bounds=self.parameter_bounds[name],
            )
        return param_dict

    def descale_phy_stat_parameters(
            self,
            phy_stat_params: torch.Tensor,
            stat_list: list,
    ) -> dict:
        """Descale routing parameters.

        Parameters
        ----------
        routing_params
            Normalized routing parameters.

        Returns
        -------
        dict
            Dictionary of descaled routing parameters.
        """
        parameter_dict = {}
        for i, name in enumerate(stat_list):
            param = phy_stat_params[:, i, :]

            parameter_dict[name] = change_param_range(
                param=param,
                bounds=self.parameter_bounds[name],
            )
        return parameter_dict

    def descale_rout_parameters(
            self,
            routing_params: torch.Tensor,
    ) -> dict:
        """Descale routing parameters.

        Parameters
        ----------
        routing_params
            Normalized routing parameters.

        Returns
        -------
        dict
            Dictionary of descaled routing parameters.
        """
        parameter_dict = {}
        for i, name in enumerate(self.routing_parameter_bounds.keys()):
            param = routing_params[:, i]

            parameter_dict[name] = change_param_range(
                param=param,
                bounds=self.routing_parameter_bounds[name],
            )
        return parameter_dict

    def forward(
            self,
            x_dict: dict[str, torch.Tensor],
            parameters: tuple[torch.Tensor],
    ) -> Union[tuple, dict[str, torch.Tensor]]:
        """Forward pass for HBV1.1p.

        Parameters
        ----------
        x_dict
            Dictionary of input forcing data.
        parameters
            Unprocessed, learned parameters from a neural network.

        Returns
        -------
        Union[tuple, dict]
            Tuple or dictionary of model outputs.
        """
        # Unpack input data.
        x = x_dict['x_phy']
        self.muwts = x_dict.get('muwts', None)

        if self.warm_up_states:
            warm_up = self.warm_up
        else:
            # No state warm up - run the full model for warm_up days.
            self.pred_cutoff = self.warm_up
            warm_up = 0

        # Unpack parameters.
        phy_dy_params, phy_static_params, routing_params = self.unpack_parameters(parameters)

        if self.routing:
            self.routing_param_dict = self.descale_rout_parameters(routing_params)

        n_grid = x.size(1)

        # Initialize model states.
        SNOWPACK = torch.zeros([n_grid, self.nmul],
                               dtype=torch.float32,
                               device=self.device) + 0.001
        MELTWATER = torch.zeros([n_grid, self.nmul],
                                dtype=torch.float32,
                                device=self.device) + 0.001
        SM = torch.zeros([n_grid, self.nmul],
                         dtype=torch.float32,
                         device=self.device) + 0.001
        SUZ = torch.zeros([n_grid, self.nmul],
                          dtype=torch.float32,
                          device=self.device) + 0.001
        SLZ = torch.zeros([n_grid, self.nmul],
                          dtype=torch.float32,
                          device=self.device) + 0.001

        phy_dy_params_dict = self.descale_phy_dy_parameters(
            phy_dy_params,
            dy_list=self.dynamic_params,
        )

        phy_static_params_dict = self.descale_phy_stat_parameters(
            phy_static_params,
            stat_list=[param for param in self.phy_param_names if param not in self.dynamic_params],
        )

        # Run the model for the remainder of simulation period.
        return self.PBM(
            x,
            [SNOWPACK, MELTWATER, SM, SUZ, SLZ],
            phy_dy_params_dict,
            phy_static_params_dict,
        )

    def PBM(
            self,
            forcing: torch.Tensor,
            states: tuple,
            phy_dy_params_dict: dict,
            phy_static_params_dict: dict,
    ) -> Union[tuple, dict[str, torch.Tensor]]:
        """Run the HBV1.1p model forward.

        Parameters
        ----------
        forcing
            Input forcing data.
        states
            Initial model states.
        full_param_dict
            Dictionary of model parameters.

        Returns
        -------
        Union[tuple, dict]
            Tuple or dictionary of model outputs.
        """
        SNOWPACK, MELTWATER, SM, SUZ, SLZ = states

        # Forcings
        P = forcing[:, :, self.variables.index('prcp')]  # Precipitation
        T = forcing[:, :, self.variables.index('tmean')]  # Mean air temp
        PET = forcing[:, :, self.variables.index('pet')]  # Potential ET

        # Expand dims to accomodate for nmul models.
        Pm = P.unsqueeze(2).repeat(1, 1, self.nmul)
        Tm = T.unsqueeze(2).repeat(1, 1, self.nmul)
        PETm = PET.unsqueeze(-1).repeat(1, 1, self.nmul)

        n_steps, n_grid = P.size()

        # Apply correction factor to precipitation
        # P = parPCORR.repeat(n_steps, 1) * P

        qsim_steps = []
        q0_steps = []
        q1_steps = []
        q2_steps = []
        aet_steps = []
        swe_steps = []
        capillary_steps = []
        recharge_steps = []
        excs_steps = []
        evapfactor_steps = []
        tosoil_steps = []
        soil_wetness_steps = []
        perc_steps = []
        sm_steps = []

        param_dict = {}
        for t in range(n_steps):
            # Get dynamic parameter values per timestep.
            for key in phy_dy_params_dict.keys():
                param_dict[key] = phy_dy_params_dict[key][t, :, :]
            for key in phy_static_params_dict.keys():
                param_dict[key] = phy_static_params_dict[key][:, :]

            (
                qsim_t,
                q0_t,
                q1_t,
                q2_t,
                aet_t,
                swe_t,
                capillary_t,
                recharge_t,
                excs_t,
                evapfactor_t,
                tosoil_t,
                soil_wetness_t,
                perc_t,
                sm_t,
                SNOWPACK,
                MELTWATER,
                SM,
                SUZ,
                SLZ,
            ) = self._step(
                Pm[t, :, :],
                Tm[t, :, :],
                PETm[t, :, :],
                SNOWPACK,
                MELTWATER,
                SM,
                SUZ,
                SLZ,
                param_dict['parBETA'],
                param_dict['parFC'],
                param_dict['parK0'],
                param_dict['parK1'],
                param_dict['parK2'],
                param_dict['parLP'],
                param_dict['parPERC'],
                param_dict['parUZL'],
                param_dict['parTT'],
                param_dict['parCFMAX'],
                param_dict['parCFR'],
                param_dict['parCWH'],
                param_dict['parBETAET'],
                param_dict['parC'],
                self.nearzero,
            )

            qsim_steps.append(qsim_t)
            q0_steps.append(q0_t)
            q1_steps.append(q1_t)
            q2_steps.append(q2_t)
            aet_steps.append(aet_t)
            swe_steps.append(swe_t)
            capillary_steps.append(capillary_t)
            recharge_steps.append(recharge_t)
            excs_steps.append(excs_t)
            evapfactor_steps.append(evapfactor_t)
            tosoil_steps.append(tosoil_t)
            soil_wetness_steps.append(soil_wetness_t)
            perc_steps.append(perc_t)
            sm_steps.append(sm_t)

        Qsimmu = torch.stack(qsim_steps, dim=0)
        Q0_sim = torch.stack(q0_steps, dim=0)
        Q1_sim = torch.stack(q1_steps, dim=0)
        Q2_sim = torch.stack(q2_steps, dim=0)
        AET = torch.stack(aet_steps, dim=0)
        SWE_sim = torch.stack(swe_steps, dim=0)
        capillary_sim = torch.stack(capillary_steps, dim=0)
        recharge_sim = torch.stack(recharge_steps, dim=0)
        excs_sim = torch.stack(excs_steps, dim=0)
        evapfactor_sim = torch.stack(evapfactor_steps, dim=0)
        tosoil_sim = torch.stack(tosoil_steps, dim=0)
        soil_wetness_sim = torch.stack(soil_wetness_steps, dim=0)
        PERC_sim = torch.stack(perc_steps, dim=0)
        SM_sim = torch.stack(sm_steps, dim=0)

        # Get the overall average
        # or weighted average using learned weights.
        if self.muwts is None:
            Qsimavg = Qsimmu.mean(-1)
        else:
            Qsimavg = (Qsimmu * self.muwts).sum(-1)

        # Run routing
        if self.routing:
            # Routing for all components or just the average.
            if self.comprout:
                # All components; reshape to [time, gages * num models]
                Qsim = Qsimmu.view(n_steps, n_grid * self.nmul)
            else:
                # Average, then do routing.
                Qsim = Qsimavg

            UH = uh_gamma(
                self.routing_param_dict['rout_a'].repeat(n_steps, 1).unsqueeze(-1),
                self.routing_param_dict['rout_b'].repeat(n_steps, 1).unsqueeze(-1),
                lenF=15,
            )
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])  # [gages,vars,time]
            UH = UH.permute([1, 2, 0])  # [gages,vars,time]
            Qsrout = uh_conv(rf, UH).permute([2, 0, 1])

            # Routing individually for Q0, Q1, and Q2, all w/ dims [gages,vars,time].
            rf_Q0 = Q0_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q0_rout = uh_conv(rf_Q0, UH).permute([2, 0, 1])
            rf_Q1 = Q1_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q1_rout = uh_conv(rf_Q1, UH).permute([2, 0, 1])
            rf_Q2 = Q2_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q2_rout = uh_conv(rf_Q2, UH).permute([2, 0, 1])

            if self.comprout:
                # Qs is now shape [time, [gages*num models], vars]
                Qstemp = Qsrout.view(n_steps, n_grid, self.nmul)
                if self.muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * self.muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else:
            # No routing, only output the average of all model sims.
            Qsim = Qsimavg
            Qs = torch.unsqueeze(Qsimavg, -1)
            Q0_rout = Q1_rout = Q2_rout = None

        if self.initialize:
            # If initialize is True, only return warmed-up storages.
            return SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            # Return all sim results.
            out_dict = {
                'streamflow': Qs,  # Routed Streamflow
                'srflow': Q0_rout,  # Routed surface runoff
                'ssflow': Q1_rout,  # Routed subsurface flow
                'gwflow': Q2_rout,  # Routed groundwater flow
                'AET_hydro': AET.mean(-1, keepdim=True),  # Actual ET
                'AET_full': AET,  # Actual ET
                'SM_full': SM_sim,  # Actual ET
                'soilwetness_full': soil_wetness_sim,  # Actual ET
                'tosoil_full': tosoil_sim,  # Actual ET
                'recharge_full': recharge_sim,  # Actual ET
                'PET_hydro': PETm.mean(-1, keepdim=True),  # Potential ET
                'SWE': SWE_sim.mean(-1, keepdim=True),  # Snow water equivalent
                'streamflow_no_rout': Qsim.unsqueeze(dim=2),  # Streamflow
                'srflow_no_rout': Q0_sim.mean(-1, keepdim=True),  # Surface runoff
                'ssflow_no_rout': Q1_sim.mean(-1, keepdim=True),  # Subsurface flow
                'gwflow_no_rout': Q2_sim.mean(-1, keepdim=True),  # Groundwater flow
                'recharge': recharge_sim.mean(-1, keepdim=True),  # Recharge
                'excs': excs_sim.mean(-1, keepdim=True),  # Excess stored water
                'evapfactor': evapfactor_sim.mean(-1, keepdim=True),  # Evaporation factor
                'tosoil': tosoil_sim.mean(-1, keepdim=True),  # Infiltration
                'percolation': PERC_sim.mean(-1, keepdim=True),  # Percolation
                'soilwater': SM_sim.mean(-1, keepdim=True),  # SM
                'capillary': capillary_sim.mean(-1, keepdim=True),  # Capillary rise
                # 'BFI': BFI_sim,  # Baseflow index
            }

            if not self.warm_up_states:
                for key in out_dict.keys():
                    if out_dict[key] is not None:
                        out_dict[key] = out_dict[key][self.pred_cutoff:, :, :]
            return out_dict
