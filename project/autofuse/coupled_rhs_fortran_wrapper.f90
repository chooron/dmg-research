PROGRAM COUPLED_RHS_ONE_STEP
  USE nrtype
  USE model_defn
  USE model_defnames
  USE multiparam
  USE multiforce
  USE multistate
  USE multi_flux
  USE model_numerix
  USE xtry_2_str_module
  USE str_2_xtry_module
  USE fuse_deriv_module
  IMPLICIT NONE

  INTEGER(I4B) :: model_id, nstate_in, i
  INTEGER(I4B) :: decision_codes(9)
  REAL(SP) :: values(37), effective, pet, dt, powlamb, maxpow
  REAL(SP), ALLOCATABLE :: s0(:), d0(:), d1(:), pred(:), pred_safe(:), heun(:), final_state(:)
  REAL(SP) :: flux0(19), flux1(19), flux_avg(19), flux_final(19), pred_flux_fixed(19), final_flux_before(19)
  REAL(SP) :: pred_errors(9), final_errors(9), state_codes(9), pred_correction(9), final_correction(9)
  REAL(SP) :: state_low(9), state_high(9), pred_low(9), pred_high(9), final_low(9), final_high(9)
  REAL(SP) :: predictor_hidden_free2a, predictor_hidden_free2b, final_hidden_free2a, final_hidden_free2b
  LOGICAL(LGT) :: bound_error

  READ(*,*) model_id
  READ(*,*) decision_codes
  READ(*,*) values
  READ(*,*) nstate_in
  ALLOCATE(s0(nstate_in), d0(nstate_in), d1(nstate_in), pred(nstate_in), pred_safe(nstate_in), heun(nstate_in), final_state(nstate_in))
  READ(*,*) s0
  READ(*,*) effective, pet, dt, powlamb, maxpow

  SMODL%MODIX = model_id
  SMODL%iRFERR = decision_codes(1)
  SMODL%iARCH1 = decision_codes(2)
  SMODL%iARCH2 = decision_codes(3)
  SMODL%iQSURF = decision_codes(4)
  SMODL%iQPERC = decision_codes(5)
  SMODL%iESOIL = decision_codes(6)
  SMODL%iQINTF = decision_codes(7)
  SMODL%iQ_TDH = decision_codes(8)
  SMODL%iSNOWM = decision_codes(9)
  CALL ASSIGN_STT()
  CALL ASSIGN_FLX()
  CALL INITFLUXES()

  MPARAM%RFERR_ADD = values(1)
  MPARAM%RFERR_MLT = values(2)
  MPARAM%RFH1_MEAN = values(3)
  MPARAM%RFH2_SDEV = values(4)
  MPARAM%RH1P_MEAN = values(5)
  MPARAM%RH1P_SDEV = values(6)
  MPARAM%RH2P_MEAN = values(7)
  MPARAM%RH2P_SDEV = values(8)
  MPARAM%MAXWATR_1 = values(9)
  MPARAM%MAXWATR_2 = values(10)
  MPARAM%FRACTEN = values(11)
  MPARAM%FRCHZNE = values(12)
  MPARAM%FPRIMQB = values(13)
  MPARAM%RTFRAC1 = values(14)
  MPARAM%PERCRTE = values(15)
  MPARAM%PERCEXP = values(16)
  MPARAM%SACPMLT = values(17)
  MPARAM%SACPEXP = values(18)
  MPARAM%PERCFRAC = values(19)
  MPARAM%FRACLOWZ = values(20)
  MPARAM%IFLWRTE = values(21)
  MPARAM%BASERTE = values(22)
  MPARAM%QB_POWR = values(23)
  MPARAM%QB_PRMS = values(24)
  MPARAM%QBRATE_2A = values(25)
  MPARAM%QBRATE_2B = values(26)
  MPARAM%SAREAMAX = values(27)
  MPARAM%AXV_BEXP = values(28)
  MPARAM%LOGLAMB = values(29)
  MPARAM%TISHAPE = values(30)
  MPARAM%TIMEDELAY = values(31)
  MPARAM%MBASE = values(32)
  MPARAM%MFMAX = values(33)
  MPARAM%MFMIN = values(34)
  MPARAM%PXTEMP = values(35)
  MPARAM%OPG = values(36)
  MPARAM%LAPSE = values(37)

  DPARAM%MAXTENS_1 = MPARAM%FRACTEN * MPARAM%MAXWATR_1
  DPARAM%MAXTENS_2 = MPARAM%FRACTEN * MPARAM%MAXWATR_2
  DPARAM%MAXFREE_1 = (1._SP - MPARAM%FRACTEN) * MPARAM%MAXWATR_1
  DPARAM%MAXFREE_2 = (1._SP - MPARAM%FRACTEN) * MPARAM%MAXWATR_2
  DPARAM%MAXTENS_1A = MPARAM%FRCHZNE * DPARAM%MAXTENS_1
  DPARAM%MAXTENS_1B = (1._SP - MPARAM%FRCHZNE) * DPARAM%MAXTENS_1
  DPARAM%MAXFREE_2A = MPARAM%FPRIMQB * DPARAM%MAXFREE_2
  DPARAM%MAXFREE_2B = (1._SP - MPARAM%FPRIMQB) * DPARAM%MAXFREE_2
  DPARAM%RTFRAC2 = 1._SP - MPARAM%RTFRAC1
  DPARAM%POWLAMB = powlamb
  DPARAM%MAXPOW = maxpow
  SELECT CASE(SMODL%iARCH2)
  CASE(iopt_tens2pll_2)
    DPARAM%QBSAT = MPARAM%QBRATE_2A * DPARAM%MAXFREE_2A + MPARAM%QBRATE_2B * DPARAM%MAXFREE_2B
  CASE(iopt_unlimfrc_2)
    DPARAM%QBSAT = MPARAM%QB_PRMS * MPARAM%MAXWATR_2
  CASE(iopt_unlimpow_2)
    DPARAM%QBSAT = MPARAM%BASERTE * (MPARAM%MAXWATR_2 / 1000._SP / MPARAM%QB_POWR) / DPARAM%POWLAMB**MPARAM%QB_POWR
  CASE(iopt_fixedsiz_2)
    DPARAM%QBSAT = MPARAM%BASERTE
  CASE DEFAULT
    DPARAM%QBSAT = MPARAM%BASERTE
  END SELECT


  state_codes = 0._SP
  state_low = 0._SP
  state_high = 0._SP
  pred_errors = 0._SP
  final_errors = 0._SP
  pred_low = 0._SP
  pred_high = 0._SP
  final_low = 0._SP
  final_high = 0._SP
  pred_correction = 0._SP
  final_correction = 0._SP
  DO i=1,nstate_in
    state_codes(i) = REAL(CSTATE(i)%iSNAME, SP)
    SELECT CASE(CSTATE(i)%iSNAME)
    CASE(iopt_TENS1A)
      state_low(i) = 1.e-9_SP * DPARAM%MAXTENS_1A; state_high(i) = DPARAM%MAXTENS_1A
    CASE(iopt_TENS1B)
      state_low(i) = 1.e-9_SP * DPARAM%MAXTENS_1B; state_high(i) = DPARAM%MAXTENS_1B
    CASE(iopt_TENS_1)
      state_low(i) = 1.e-9_SP * DPARAM%MAXTENS_1; state_high(i) = DPARAM%MAXTENS_1
    CASE(iopt_FREE_1)
      state_low(i) = 1.e-9_SP * DPARAM%MAXFREE_1; state_high(i) = DPARAM%MAXFREE_1
    CASE(iopt_WATR_1)
      state_low(i) = 1.e-9_SP * MPARAM%MAXWATR_1; state_high(i) = MPARAM%MAXWATR_1
    CASE(iopt_TENS_2)
      state_low(i) = 1.e-9_SP * DPARAM%MAXTENS_2; state_high(i) = DPARAM%MAXTENS_2
    CASE(iopt_FREE2A)
      state_low(i) = 1.e-9_SP * DPARAM%MAXFREE_2A; state_high(i) = DPARAM%MAXFREE_2A
    CASE(iopt_FREE2B)
      state_low(i) = 1.e-9_SP * DPARAM%MAXFREE_2B; state_high(i) = DPARAM%MAXFREE_2B
    CASE(iopt_WATR_2)
      state_low(i) = 1.e-9_SP * MPARAM%MAXWATR_2; state_high(i) = MPARAM%MAXWATR_2
    END SELECT
  END DO
  FRACSTATE_MIN = 1.e-9_SP
  SOLUTION_METHOD = EXPLICIT_EULER
  CURRENT_DT = dt
  MFORCE%PPT = effective
  MFORCE%PET = pet
  MFORCE%TEMP = 0._SP
  CALL XTRY_2_STR(s0, MSTATE)
  M_FLUX%EFF_PPT = effective
  d0 = FUSE_DERIV(s0)
  CALL PACK_FLUX(M_FLUX, flux0)

  pred = s0 + dt * d0
  BSTATE = MSTATE
  CALL XTRY_2_STR(pred, ESTATE)
  IF (SMODL%iARCH2.NE.iopt_tens2pll_2) ESTATE%FREE_2B = 0._SP
  predictor_hidden_free2a = ESTATE%FREE_2A
  predictor_hidden_free2b = ESTATE%FREE_2B
  CALL FIX_STATES(dt, bound_error)
  CALL STR_2_XTRY(ESTATE, pred_safe)
  CALL PACK_FLUX(M_FLUX, pred_flux_fixed)
  CALL PACK_ERRORS(M_FLUX, pred_errors)
  CALL CLASSIFY_BOUNDS(pred, state_low, state_high, pred_low, pred_high)
  pred_correction(1:nstate_in) = pred_safe - pred

  M_FLUX%EFF_PPT = effective
  d1 = FUSE_DERIV(pred_safe)
  CALL PACK_FLUX(M_FLUX, flux1)
  heun = s0 + 0.5_SP * dt * (d0 + d1)
  flux_avg = 0.5_SP * (flux0 + flux1)

  CALL UNPACK_FLUX(flux_avg, M_FLUX)
  BSTATE = MSTATE
  CALL XTRY_2_STR(heun, ESTATE)
  IF (SMODL%iARCH2.NE.iopt_tens2pll_2) ESTATE%FREE_2B = 0._SP
  final_hidden_free2a = ESTATE%FREE_2A
  final_hidden_free2b = ESTATE%FREE_2B
  CALL FIX_STATES(dt, bound_error)
  CALL STR_2_XTRY(ESTATE, final_state)
  CALL PACK_ERRORS(M_FLUX, final_errors)
  CALL CLASSIFY_BOUNDS(heun, state_low, state_high, final_low, final_high)
  final_correction(1:nstate_in) = final_state - heun
  final_flux_before = flux_avg
  CALL PACK_FLUX(M_FLUX, flux_final)

  WRITE(*,'(A,I0)') 'MODEL ', model_id
  CALL WRITE_VECTOR('STATE0', s0)
  CALL WRITE_VECTOR('D0', d0)
  CALL WRITE_VECTOR('FLUX0', flux0)
  CALL WRITE_VECTOR('PREDICTOR', pred)
  CALL WRITE_VECTOR('PREDICTOR_SAFE', pred_safe)
  CALL WRITE_VECTOR('D1', d1)
  CALL WRITE_VECTOR('FLUX1', flux1)
  CALL WRITE_VECTOR('HEUN_RAW', heun)
  CALL WRITE_VECTOR('FLUX_AVG', flux_avg)
  CALL WRITE_VECTOR('STATE1', final_state)
  CALL WRITE_VECTOR('FLUX_FINAL', flux_final)
  CALL WRITE_VECTOR('PREDICTOR_FLUX_FIXED', pred_flux_fixed)
  CALL WRITE_VECTOR('PREDICTOR_ERRORS', pred_errors)
  CALL WRITE_VECTOR('PREDICTOR_STATE_CORRECTION', pred_correction)
  CALL WRITE_VECTOR('PREDICTOR_LOWER_VIOLATION', pred_low)
  CALL WRITE_VECTOR('PREDICTOR_UPPER_VIOLATION', pred_high)
  CALL WRITE_VECTOR('FINAL_FLUX_BEFORE_FIX', final_flux_before)
  CALL WRITE_VECTOR('FINAL_ERRORS', final_errors)
  CALL WRITE_VECTOR('FINAL_STATE_CORRECTION', final_correction)
  CALL WRITE_VECTOR('FINAL_LOWER_VIOLATION', final_low)
  CALL WRITE_VECTOR('FINAL_UPPER_VIOLATION', final_high)
  CALL WRITE_VECTOR('STATE_CODES', state_codes)
  CALL WRITE_VECTOR('STATE_LOWER_BOUNDS', state_low)
  CALL WRITE_VECTOR('STATE_UPPER_BOUNDS', state_high)
  WRITE(*,'(A,1X,ES24.16)') 'PREDICTOR_HIDDEN_FREE2A', predictor_hidden_free2a
  WRITE(*,'(A,1X,ES24.16)') 'PREDICTOR_HIDDEN_FREE2B', predictor_hidden_free2b
  WRITE(*,'(A,1X,ES24.16)') 'FINAL_HIDDEN_FREE2A', final_hidden_free2a
  WRITE(*,'(A,1X,ES24.16)') 'FINAL_HIDDEN_FREE2B', final_hidden_free2b

CONTAINS

  SUBROUTINE PACK_FLUX(source, target)
    TYPE(FLUXES), INTENT(IN) :: source
    REAL(SP), INTENT(OUT) :: target(19)
    target = (/ source%EFF_PPT, source%SATAREA, source%EVAP_1A, source%EVAP_1B, source%EVAP_1, &
      source%RCHR2EXCS, source%TENS2FREE_1, source%QPERC_12, source%QINTF_1, source%OFLOW_1, &
      source%QSURF, source%EVAP_2, source%TENS2FREE_2, source%QBASE_2A, source%QBASE_2B, &
      source%QBASE_2, source%OFLOW_2A, source%OFLOW_2B, source%OFLOW_2 /)
  END SUBROUTINE PACK_FLUX

  SUBROUTINE UNPACK_FLUX(source, target)
    REAL(SP), INTENT(IN) :: source(19)
    TYPE(FLUXES), INTENT(OUT) :: target
    target%EFF_PPT = source(1); target%SATAREA = source(2); target%EVAP_1A = source(3)
    target%EVAP_1B = source(4); target%EVAP_1 = source(5); target%RCHR2EXCS = source(6)
    target%TENS2FREE_1 = source(7); target%QPERC_12 = source(8); target%QINTF_1 = source(9)
    target%OFLOW_1 = source(10); target%QSURF = source(11); target%EVAP_2 = source(12)
    target%TENS2FREE_2 = source(13); target%QBASE_2A = source(14); target%QBASE_2B = source(15)
    target%QBASE_2 = source(16); target%OFLOW_2A = source(17); target%OFLOW_2B = source(18)
    target%OFLOW_2 = source(19)
  END SUBROUTINE UNPACK_FLUX

  SUBROUTINE PACK_ERRORS(source, target)
    TYPE(FLUXES), INTENT(IN) :: source
    REAL(SP), INTENT(OUT) :: target(9)
    target = (/ source%ERR_TENS_1A, source%ERR_TENS_1B, source%ERR_TENS_1, source%ERR_FREE_1, source%ERR_WATR_1, &
      source%ERR_TENS_2, source%ERR_FREE_2A, source%ERR_FREE_2B, source%ERR_WATR_2 /)
  END SUBROUTINE PACK_ERRORS
  SUBROUTINE CLASSIFY_BOUNDS(candidate, lower, upper, lower_flag, upper_flag)
    REAL(SP), INTENT(IN) :: candidate(:), lower(:), upper(:)
    REAL(SP), INTENT(OUT) :: lower_flag(:), upper_flag(:)
    lower_flag = 0._SP; upper_flag = 0._SP
    DO i=1,SIZE(candidate)
      IF (candidate(i).LT.lower(i)) lower_flag(i)=1._SP
      IF (candidate(i).GT.upper(i)) upper_flag(i)=1._SP
    END DO
  END SUBROUTINE CLASSIFY_BOUNDS

  SUBROUTINE WRITE_VECTOR(label, vector)
    CHARACTER(LEN=*), INTENT(IN) :: label
    REAL(SP), INTENT(IN) :: vector(:)
    WRITE(*,'(A,1X,*(ES24.16,1X))') TRIM(label), vector
  END SUBROUTINE WRITE_VECTOR

END PROGRAM COUPLED_RHS_ONE_STEP
