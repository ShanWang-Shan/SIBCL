#!/bin/bash
declare -A file_dict

file_dict['2017-08-04-V2-Log3-info_files']='EZbpEOaKIetMjAKUOGCSEjcBGQSwizpckJPKlr4s1dH0sA?e=yaoKqg'
file_dict['2017-08-04-V2-Log4-info_files']='EYOqUb_1a8tDsMBoAokV6S8BzjjwRUrlEPJB_njE5VGKDA?e=Gh2quy'
file_dict['2017-08-04-V2-Log5-info_files']='EYtT85HV3fRFiAFdZWQ7zLMB5-2sG26MnZEbXiGRpnTxXQ?e=vlJdbg'
file_dict['2017-08-04-V2-Log6-info_files']='EeDaibG69SxNjNI-DSzQHcMBUOEhiQEGOpqXi0hhyaMltg?e=f41sS3'
file_dict['2017-10-26-V2-Log3-info_files']='EYFArPf4IOFAgO9ZP-z8tEgBgQdLunDt97nCTLsAYxROUQ?e=EM5y4p'
file_dict['2017-10-26-V2-Log4-info_files']='EfkruM3-WaVIkvRt7yaMjQcBixEY9mxKUa6EPvllcbcVmQ?e=mu7yP4'
file_dict['2017-10-26-V2-Log5-info_files']='ESd4FRPRyGtCtbgIUa3wM_kBKNZKfTlU7KPMx1knG1b1jw?e=qciLUz'
file_dict['2017-10-26-V2-Log6-info_files']='EfIfRoI7S5xBmr3GJvYiBMkBwDKetYgXpJrD1kyLUZphew?e=7PL8B9'

file_dict['2017-08-04-V2-Log3-pcd']='EXJERyuFiGhAiXaxMhFIu9cBmVeFCifTQFkgzxKrXb3zkQ?e=24eHJ9'
file_dict['2017-08-04-V2-Log4-pcd']='EcDHVG_9YpZNlfL3bLp-180BV-YhPgeUjTWbDgjt4PpCtg?e=fgqotg'
file_dict['2017-08-04-V2-Log5-pcd']='EVk_8Ah6q_hClNyEaQ7vSmYBYBysVyGZzGJmiSHMte78qg?e=qUalA1'
file_dict['2017-08-04-V2-Log6-pcd']='EYMPio1gSVFAlGGOKzc9Ff8Bl-KeChtVB8Z6aiyfp1uSMg?e=qX9DxP'
file_dict['2017-10-26-V2-Log3-pcd']='EW_5iCAmfpBIlatZ7TD7FuoB_8LATyybgjdYf4THWxDNUw?e=l99gsl'
file_dict['2017-10-26-V2-Log4-pcd']='EXq8jhCk-odBiXlRuIBu71YBFhXbmcSmXmNOITD2yEOEiA?e=W4vXof'
file_dict['2017-10-26-V2-Log5-pcd']='EWCL-kr6EIhHiEShZpsi2JwB5e4FxyJh01sUDKkYLpsg7Q?e=zBdIVT'
file_dict['2017-10-26-V2-Log6-pcd']='Ecmi_tkvzWdIsS4hV1A-4xMBvGx1cysWe5HxIueaYnoGuA?e=1g9dlE'

for key in "${!file_dict[@]}"; do
  link=${file_dict[$key]}
  name=$key'.tar.gz'
  dir=${key: 0: 18}'/'${i: 19}
        if [ ! -d ${i: 0: 18} ]; then
                mkdir ${i: 0: 18}
        fi
        if [ ! -d $dir ]; then
                mkdir $dir
        fi
	echo "Downloading: "$key
        wget --no-check-certificate 'https://anu365-my.sharepoint.com/:u:/g/personal/u7094434_anu_edu_au/'$link'&download=1'
        mv $link'&download=1' $name
        tar -zxvf $name -C $dir
        rm $name
done
