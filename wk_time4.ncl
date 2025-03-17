;---------------------------------------------
; wkSpaceTime_4.ncl
;---------------------------------------------
;
; These files are loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
;
; This file still has to be loaded manually
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/diagnostics_cam.ncl"

  case     = "OLR 15N-15S" ; case name
  diro     = "./" 

  latN     = 15
  latS     = -latN       ; symmetric about the equator

  nDayWin  = 96      ; Wheeler-Kiladis [WK] temporal window length (days)
  nDaySkip = 30      ; time (days) between temporal windows [segments]
                     ; negative means there will be overlapping 
                     ; temporal segments
  
  opt      = True
  opt@debug= True
  opt@pltType     = "png"   ; send graphics to PNG file
  opt@cnLinesOn   = False   ; turn off contour lines
                     ; specify the plot intervals

  spd      = 1       ; samples per day 
  var      = "olr"  ; user specified 



  f        = addfile("/Users/xpji/spacetime_filter/code_ncl/olr.day.mean_17.nc", "r") 
  
  time= f->time
	lat = f->lat({latS:latN})
	lon = f->lon
  VAR = f->olr(time|:, {lat|latS:latN}, lon|:)   ; extract desire latitude band

  printVarSummary(VAR)

  nMsg     = num(ismissing(VAR))  ; check for _FillValue
  if (nMsg.gt.0) then
      print("missing values present: not allowed: nMsg="+nMsg) 
      exit
  end if
 
                     ; calculation and create plots
  wkSpaceTime (VAR, diro, case, var             \
              ,latN, spd, nDayWin, nDaySkip, opt  )

