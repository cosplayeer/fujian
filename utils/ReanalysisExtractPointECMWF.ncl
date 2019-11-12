load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "/usr/ncl/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"
load "/usr/ncl/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "/usr/ncl/lib/ncarg/nclscripts/csm/contributed.ncl"
 
file1 = "/home/meteodyn/DYP/MOS-X-WQS/Seasonal_forecast_daily_sfc_20190101.grib"
f = addfile(file1,"r")

lat0 = f->g0_lat_2
lon0 = f->g0_lon_3

foutf = "output.csv"

;get min and max value of long and lat automatically, and replace it.
;MERRA2:
;   xi = fspan(-180,179.375,576)
;   yi = fspan(-90, 90,361)
;FNL: 0.25 * 0.25 degrees:
;   xi = fspan(0,359.75,1440)
;   yi = fspan(-90, 90,721)
;ERAI: 0.75 * 0.75 degrees:
;    xi = fspan(0,359.25,480)
;    yi = fspan(-90, 90,241)
   xi = fspan(0,359,360)
   yi = fspan(-90, 90,181)
   xo = (/ 119.1930 /)
   yo = (/ 25.2081 /)

;    timehours=f[i]->initial_time0_hours
;    time=cd_calendar(timehours,-3)
    U10=f->10U_GDS0_SFC(:,:,:,:)   ; ( initial_time0_hours, g0_lat_1, g0_lon_2 )
    V10=f->10V_GDS0_SFC(:,:,:,:)
    ;PSFC ; Surface pressure
    ; PSFC=f[i]->SP_GDS0_SFC(:,:,:)
    ; new variables
    ;T2
    ; T2=f[i]->2T_GDS0_SFC(:,:,:)
    ;QV80M  ; specific humidity
;    QV2Ms = f[i]->SPFH_P0_L103_GLL0(0,:,:)
;    QV80Ms = f[i]->SPFH_P0_L103_GLL0(1,:,:)
    ;RH
;    QV2M = f[i]->RH_P0_L103_GLL0(:,:) ; 
    ;SST 
    ; SST = f[i]->SSTK_GDS0_SFC(:,:,:)

    ; interpolation
    U10point = linint2_points(xi, yi, U10, True, xo, yo, 0)
    V10point = linint2_points(xi, yi, V10, True, xo, yo, 0)

    ; interpolation todo
    ; convert from u, v to wind and winddir
    wind10 = sqrt(U10point^2 + V10point^2) ;51 x 480
    ; winddir10 = wind_direction(U10point, V10point,0)

    wind10T = transpose(wind10)
    printVarSummary(wind10T)
    ;output write csv file.
    initial_time0="2019010100"
    _timehours=f->forecast_time1
    _timehours@units="hours since 2019-01-01 00:00:0.0"
    timehours = cd_calendar(_timehours,-3) ;YYYYmmddhh

    itime=0
    numbertime = dimsizes(timehours)
    siymdh_list = new((/numbertime,1/),string)

    do while( itime .lt. numbertime)
        timetemp = timehours(itime)
        stringiymdh = tostring(timetemp)
        chariymdh = stringtochar(stringiymdh)
        
        stringYYYY= tostring(chariymdh(0:3))
        stringMM= tostring(chariymdh(4:5))
        stringDD= tostring(chariymdh(6:7))
        stringhh= tostring(chariymdh(8:9))
        siymdh_list(itime,0) = stringYYYY + "-" + stringMM + "-" + stringDD + " " + stringhh + ":00:00"
        itime = itime + 1
    end do

    ; sheet 
    header = (/"TimeInfo, WindSpeedVar1, WindSpeedVar2, WindSpeedVar3, WindSpeedVar4, WindSpeedVar5, WindSpeedVar6, WindSpeedVar7, WindSpeedVar8, WindSpeedVar9, WindSpeedVar10," + "WindSpeedVar11, WindSpeedVar12, WindSpeedVar13, WindSpeedVar14, WindSpeedVar15, WindSpeedVar16, WindSpeedVar17, WindSpeedVar18, WindSpeedVar19, WindSpeedVar20" + "WindSpeedVar21, WindSpeedVar22, WindSpeedVar23, WindSpeedVar24, WindSpeedVar25, WindSpeedVar26, WindSpeedVar27, WindSpeedVar28, WindSpeedVar29, WindSpeedVar30" + "WindSpeedVar31, WindSpeedVar32, WindSpeedVar33, WindSpeedVar34, WindSpeedVar35, WindSpeedVar36, WindSpeedVar37, WindSpeedVar38, WindSpeedVar39, WindSpeedVar40," + "WindSpeedVar41, WindSpeedVar42, WindSpeedVar43, WindSpeedVar44, WindSpeedVar45, WindSpeedVar46, WindSpeedVar47, WindSpeedVar48, WindSpeedVar49, WindSpeedVar50, WindSpeedVar51"/)
    hlist = [/header/]
    alist = [/siymdh_list, wind10T(0,:,0), wind10T(0,:,1), wind10T(0,:,2), wind10T(0,:,3), wind10T(0,:,4), wind10T(0,:,5), wind10T(0,:,6), wind10T(0,:,7), wind10T(0,:,8), wind10T(0,:,9), wind10T(0,:,10), wind10T(0,:,11), wind10T(0,:,12), wind10T(0,:,13), wind10T(0,:,14), wind10T(0,:,15), wind10T(0,:,16), wind10T(0,:,17), wind10T(0,:,18), wind10T(0,:,19), wind10T(0,:,20), wind10T(0,:,21), wind10T(0,:,22), wind10T(0,:,23), wind10T(0,:,24), wind10T(0,:,25), wind10T(0,:,26), wind10T(0,:,27), wind10T(0,:,28), wind10T(0,:,29), wind10T(0,:,30),wind10T(0,:,31), wind10T(0,:,32), wind10T(0,:,33), wind10T(0,:,34), wind10T(0,:,35), wind10T(0,:,36), wind10T(0,:,37), wind10T(0,:,38), wind10T(0,:,39), wind10T(0,:,40), wind10T(0,:,41), wind10T(0,:,42), wind10T(0,:,43), wind10T(0,:,44), wind10T(0,:,45), wind10T(0,:,46), wind10T(0,:,47), wind10T(0,:,48), wind10T(0,:,49), wind10T(0,:,50)/]
    ; alist = [/siymdh_list /]
    write_table(foutf, "w", hlist, "%s")
    write_table(foutf, "a", alist, "%s, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f，%9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f，%9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f，%9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f，%9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f, %9.3f")
    ; alist = [/siymdh_list, wind10list /]
    ; write_table(foutf,"a",alist, "%s, %9.3f, %9.3f")