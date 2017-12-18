###############################################################################
# file      : tools.inc
# trait     : 
# author    : xwfang
# history   :
# init 2005-07-02
# ...
# 2005-10-12 stop_proc <proc_name> [interval]
# 
# 
# 
###############################################################################

###############################################################################
# desc		:
# input		:
# output	:
# return	:
###############################################################################
is_digit()
{
    [ "$#" -eq 1 ] || return 1 # is blank
    
    #
    [ "-$1" == '-0' ] && return 0 # is zero
    
    local -i digit
    let digit=$1 2>/dev/null # else is numeric!
    return $?
}

###############################################################################
# desc		:
# input		:
# output	:
# return	:
###############################################################################
lower_case()
{    
    echo $1 | tr 'A-Z' 'a-z'
}

upper_case()
{    
    echo $1 | tr 'a-z' 'A-Z'
}

###############################################################################
# desc		:
# input		:
# output	:
# return	:
###############################################################################
is_absolute_path()
{
	#
	if [ $# -ne 1 ]
	then
	{	
		echo "Usage:$0 <path>"
		return 1
	}
	fi
	
	#
	typeset number=`echo $1 | sed -n "s/^\///gp" | wc -l`	
	if [ $number -ne 1 ]
	then
	{
		return 1
	}
	fi
	return 0
}

###############################################################################
# desc		:
# input		:
# output	:
# return	:
###############################################################################
get_ip() 
{
    ifconfig $1 2>/dev/null| grep "inet addr" | awk -F':' '{print $2}' | awk '{print $1}'
    return 0
}

###############################################################################
# desc		:
# input		:
# output	:
# return	:
###############################################################################
is_ip()
{
	if [ $# -ne 1 ]
	then
	{
		echo "Usage:$0 <string>"
		return 1
	}
	fi

	typeset ip=$1
	
	#
	typeset ipflag
	ipflag=`echo ${ip} | awk -F. '{ if ( NF==4 && ( $1 >= 0 && $1 < 256 ) 
	&& ( $2 >= 0 && $2 < 256 ) &&  ( $3 >= 0 && $3 < 256 ) && ( $4 >= 0 
	&& $4 < 256 )  ) print "ok" }'`
	
	#
	if [ "-${ipflag}" = "-ok" ]
	then
	{
		return 0
	}
	fi
	
	return 1
}


###############################################################################
# desc		:
# input		:
# output	:
###############################################################################
time_string()
{
	echo `date "+%Y-%m-%d %H:%M:%S"`
	return 0
}

date_string()
{
	echo `date "+%Y%m%d"`
	return 0
}

yesterday_date_string()
{
    #
	echo `date -d-1day +%Y%m%d`
	return 0
}

yesterday_month_string()
{
    #
	echo `date -d-1day +%Y%m`
	return 0
}

###############################################################################
# desc		:
# input		:
# output	:
###############################################################################
rchg_pwd()
{
	if [ $# -ne 3 ]
	then
	{
		echo "Error:wrong parameter,$@"
		echo "Usage:$0 <oldpasswd> <newpasswd> <ip>"
		return 1
	}
	fi	
		
	typeset oldpasswd=$1
	typeset newpasswd=$2
	typeset ip=$3
		
	#
		
	echo "Start change remote $ip passwd ........................"
	#
/usr/bin/expect <<EOF
    set timeout 10
    spawn /usr/local/bin/ssh $ip -p36000 -lroot
    expect "assword:"
    send "${oldpasswd}\r"
    expect "#"
    send "passwd\r"
    expect "New password:"
    send "${newpasswd}\r"
    expect "Re-enter new password:"
    send "${newpasswd}\r"
    expect "#"
    send "exit\r"
    expect eof
EOF
	return 0
}

###############################################################################
# desc		:
# input		:
# output	:
###############################################################################
update_file()
{
	if [ $# -ne 6 ]
	then
	{
		echo "Error:wrong parameter,$@"
		echo "Usage:$0 <filename> <user> <passwd> <ip> <port> <dir>"
		return 1
	}
	fi	
	typeset filename=$1
	typeset user=$2
	typeset passwd=$3
	typeset ip=$4
	typeset port=$5
	typeset dir=$6
	
	#
	
	#
	typeset prompt="\$"
    if [ "${user}" = "root" ]
    then
    {
        prompt="#"
    }
    fi
	
	echo "Start update file ${filename} to ${ip}:${dir}........................"
	#
/usr/bin/expect <<EOF
    set timeout 3600
    spawn scp -p ${filename} ${user}@${ip}#${port}:${dir}
    expect "password"
    send "${passwd}\r"
    expect "100%"
    spawn ssh $user@${ip}#${port}
    expect "password"
    send "${passwd}\r"
    expect "${prompt}"
    send "ls -l ${dir}/${filename##*/}\r"
    expect "${prompt}"
    send "exit\r"
    expect eof
EOF
	return 0
}


###############################################################################
# desc		:
# input		:
# output	:
###############################################################################
update_binfile()
{
	if [ $# -ne 6 ]
	then
	{
		echo "Error:wrong parameter,$@"
		echo "Usage:$0 filename user passwd ip port dir"
		return 1
	}
	fi	
	typeset filename=$1
	typeset user=$2
	typeset passwd=$3
	typeset ip=$4
	typeset port=$5
	typeset dir=$6
	
	update_file ${filename} ${user} ${passwd} ${ip} ${port} ${dir}
	
	#
	typeset prompt="\$"
    if [ "${user}" = "root" ]
    then
    {
        prompt="#"
    }
    fi
    
    echo "${prompt}"
	
	#
	echo "Set file exec attribute.............................................."
/usr/bin/expect <<EOF
    set timeout 3600
    spawn ssh $user@${ip}#${port}
    expect "password"
    send "${passwd}\r"
    expect "${prompt}"
    send "chmod a+x $dir/${filename##*/}\r"
    expect "${prompt}"
    send "ls -l $dir/${filename##*/}\r"
    expect "${prompt}"
    send "exit\r"
    expect eof
EOF
	return 0
}

###############################################################################
# desc		:
# input		:
# output	:
###############################################################################
remote_exec()
{
    if [ $# -ne 5 ]
	then
	{
		echo "Error:wrong parameter,$@"
		echo "Usage:$0 <cmd>  <ip> <port> <user> <passwd>"
		return 1
	}
	fi
	
	typeset cmd=$1	
	typeset ip=$2
	typeset port=$3
	typeset user=$4
	typeset passwd=$5

    /usr/bin/expect <<- EOF
        set timeout 60
        spawn ssh -p${port} -l${user} ${ip} ${cmd}
        expect "password"
        send "${passwd}\r"
        
        expect eof
	EOF
        
}

###############################################################################
# desc		:
# input		:
# output	:
###############################################################################
get_log()
{
	if [ $# -ne 7 ]
	then
	{
		echo "Error:wrong parameter,$@"
		echo "Usage:$0 user ip port srcdir filename destdir"
		return 1
	}
	fi
		
	typeset user=${1}
	typeset ip=${2}
	typeset port=${3}
	typeset srcdir=${4}
	typeset filename=${5}
	typeset destdir=${6}
	typeset passwd=${7}
			
	#${user} ${ip} ${port} ${srcdir} ${filename} ${destdir}
	
	#scp ${user}@${ip}#${port}:${srcdir}/${filename} ${destdir}
/usr/bin/expect <<EOF
	set timeout 3600
    spawn scp ${user}@${ip}#${port}:${srcdir}/${filename} ${destdir}
    expect "password"
    send "${passwd}\r"
    expect eof
EOF
	
	#
	rename log log.${ip} ${destdir}/*.log  2>/dev/null
	
	return 0
}

###############################################################################
# desc		:
# input		:
# output	:
###############################################################################
get_my_log()
{	
	if [ $# -ne 1 ]
	then
	{
		echo "Usage:$0 <ip>"
		return 1
	}
	fi
	
	typeset user=${USER}
	typeset ip=${1}
	typeset port=${PORT}
	typeset srcdir=${SRCDIR}
	typeset filename="${FILENAME}"
	#typeset file_date=`time_string`
	#typeset files=`ls "${filename}" 2>"/tmp/getlog_${file_date}_$$.log"`
	typeset destdir=${DESTDIR}
	typeset passwd=${PASSWD}
	if [ "-${destdir}" = "-" ]
	then
	{
		echo "Warning:use default dest dir ."
		destdir="."
	}
	fi
	
	echo "Info:get paycent log ${filename} from ${ip} ........................"

	get_log ${user} ${ip} ${port} ${srcdir} ${filename} ${destdir} ${passwd}

	
	return 0
}

###############################################################################
# desc		:ͣ
# input		:
# output	:
###############################################################################
stop_proc()
{
	if [ $# -lt 1 ]
	then
	{
		echo "Usage:$0 <proc_name> [interval]"
		return 1
	}
	fi
	
	typeset proc_name=$1
	typeset interval=$2
	
	#
	if [ "-${interval}" = "-" ]
	then
	{
		interval=1		
	}
	fi
		
	ps -ef | grep "${proc_name}" | grep -v "grep"
	#ps -fC "${proc_name}" 
	echo "Kill ${proc_name} ..."
	
	#count ${proc_name}
	
	#typeset proc_num=$(ps -ef | grep "${proc_name}" | grep -v "grep" -c)
	typeset proc_num=$(ps -fC "${proc_name}" | grep "${proc_name}" | wc -l)
	while [ ${proc_num} -gt 0 ]
	do
	    killall -9 ${proc_name}
	    sleep ${interval}
	    #proc_num=$(ps -ef | grep "${proc_name}" | grep -v "grep" -c)
	    proc_num=$(ps -fC "${proc_name}" | grep "${proc_name}" | wc -l)
	done
	echo "Stop ${proc_name} ok."	
}

###############################################################################
# desc		:
# input		:
# output	:
###############################################################################
restart_proc()
{
	if [ $# -lt 2 ]
	then
	{
		echo "Usage:$0 <proc_name> <start_path> [interval]"
		return 1
	}
	fi
	
	typeset proc_name=$1
	typeset start_path=$2	
	typeset interval=$3
	
	stop_proc ${proc_name} ${interval}
	
	echo "Restart ${proc_name} ..."
	${start_path}
	if [ $? -ne 0 ]
	then
    {
        echo "Error:restart ${proc_name} error."
        return 1
    }
    fi
	ps -ef | grep "${proc_name}" | grep -v "grep"
}

###############################################################################
# desc		:
#            
# input		:
# output	:
###############################################################################
keep_proc()
{
	if [ $# -lt 2 ]
	then
	{
		echo "Usage:$0 <proc_name> <start_path> [interval] [log_path]"
		return 1
	}
	fi
	
	typeset proc_name=$1 
	typeset start_path=$2 
	typeset interval=$3 
	typeset log_path=$4 
	typeset mobileno=$5 #
	
	#
	if [ "-${interval}" = "-" ]
	then
	{
		interval=60				#
	}
	fi
	
	if [ "-${log_path}" = "-" ]
	then
	{
		log_path="/tmp/keeper.log"		#
	}
	fi
	
	if [ "-${mobileno}" = "-" ]
	then
	{
	    mobileno="13570850050" #
	}
    fi
	
	typeset time_str=`time_string`
	echo "[${time_str}]Info: $0 monitor process ${proc_name},set interval " \
	"${interval},start path is ${start_path}." >> "${log_path}"
	
	while [ 1 ]
	do	    
	    typeset -i proc_num=$(ps -fC "${proc_name}" | grep "${proc_name}" | wc -l) #-o args= 
	    	    
	    time_str=`time_string`
	    #ip_addr=`hostname --ip-address`
	    echo "[${time_str}]Info:${proc_name} process number is ${proc_num}" >> "${log_path}"
	    
	    if [ ${proc_num} -eq 0 ]
	    then
    	{    		
    		echo "[${time_str}]Warning: ${proc_name} is not runing,restart it," \
    		"${start_path}"��>> "${log_path}"
    		
    		#send_sms "${mobileno}" "[${time_str}@${ip_addr}]Warning: ${proc_name} is not runing, restart it"
    		
    		${start_path}
    		if [ $? -ne 0 ]
        	then
            {
                echo "[${time_str}]Error: restart ${proc_name} error." >> "${log_path}"
                #send_sms "${mobileno}" "[${time_str}@${ip_addr}]Error: restart ${proc_name} error."
            }
            fi
    	}
		else
		{
			:
			#
#			typeset hm=`date "+%H%M"`
#			if [ "${hm}" = "1440" -o "${hm}" = "1441" ]
#		    then
#	        {
#	            send_sms "${mobileno}" "[${time_str}@${ip_addr}]Info:${proc_name} is runing"
#	        }
#		    fi
		}
    	fi
    	
	    sleep ${interval}	    
	done
		
}

###############################################################################
# desc		:
# input		:
# output	:
###############################################################################
get_file_size()
{
	if [ $# -ne 1 ]
	then
	{
		echo "Usage:$0 <filepath>"
		return 1
	}
	fi
	typeset filepath=$1
	typeset filesize
	filesize=`du -sk ${filepath} | awk '{print $1}'`
	echo ${filesize}
	return 0
}

###############################################################################
# desc		:
# input		:
# output	:
###############################################################################
save_yesterday_log()
{
    if [ $# -lt 1 ]
	then
	{
		echo "Usage:$0 <filepath> [<destdir>] [tar] [del]"
		return 1
	}
	fi
	
	typeset yesterday=`yesterday_date_string`
	typeset log_file=$1
	typeset dest_dir=$2
	typeset tar_flag=$3
	typeset del_flag=$4
	
	if [ "-${dest_dir}" = "-" ]
	then
    {
        #
        dest_dir="${log_file%/*}"
    }
    fi
    
#    : <<- EOF
#    while getopt td OPTION
#    do 
#        echo "$OPTION away"
#        case $OPTION
#        in 
#           t) tar_flag=1
#           ;;
#           d) del_flag=1
#           ;; 
#           \?) echo "Warning:illegal option ${OPTION}"
#           ;;
#        esac
#    done
#    EOF
	
	#
	yesterday_file="${dest_dir}/${log_file##*/}.${yesterday}"
	
	cp ${log_file} ${yesterday_file}
	if [ $? -ne 0 ]
	then
    {
        echo "Error:copy file failed, source=${log_file}, dest=${yesterday_file}"
        return 1
    }
	fi
	
	: > ${log_file}
	
	#
	typeset month=`yesterday_month_string`
	typeset month_tar_file="${dest_dir}/${log_file##*/}.${month}.tar"
	
	echo "tar_flag=${tar_flag}"
	
	if [ "-${tar_flag}" = "-tar" ]
	then
	{
	    gunzip ${month_tar_file}.gz	 2>/dev/null   
	    tar -uvPf ${month_tar_file} ${yesterday_file}
	    if [ $? -ne 0 ]
    	then
        {
            echo "Error:tar file failed,command is tar -uvf " \
            "${month_tar_file} ${yesterday_file}"
            return 1
        }
        fi
        gzip ${month_tar_file}
	    
	    if [ "-${del_flag}" = "-del" ]
    	then
    	{
    	    rm ${yesterday_file}
    	}
        fi	    
	}
    fi	
	
    return 0
}

###############################################################################
# desc		:
# input		:
# output	:
###############################################################################
send_sms()
{
    if [ $# -lt 2 ]
	then
	{
		echo "Usage:$0 <destno> <content> [<station>]"
		return 1
	}
	fi
	
	destno_list=$1
	content=$2
	station=$3
	if [ "-${station}" = "-" ]
	then
	{
		station="gdgmcc"
	}
	fi
	
	        
    {   
        for destno in ${destno_list}
        do
        {         
            #      
            echo "SRCNO=1700&DESNO=${destno}&GSMSTATION=${station}" \
            "&SMTYPE=6&CONTENT=${content}"        
            sleep 1 #  
        }
        done
        
        echo "^]"
        echo "quit"
        #
    } | telnet 172.16.53.15 26062    
    
}

alarm_mobile()
{
    typeset proc_name=$1
	typeset mobileno=$2 #
	
	if [ "-${mobileno}" = "-" ]
	then
	{
	    mobileno="13570850050" #
	}
    fi
	
    typeset -i proc_num=$(ps -fC "${proc_name}" | grep "${proc_name}" | wc -l)
	    	    
    time_str=`time_string`
    ip_addr=`hostname --ip-address`
    
    if [ ${proc_num} -eq 0 ]
    then
	{
	    echo "[${time_str}]Warning: ${proc_name} is not runing" #>> "${log_path}"
	    send_sms "${mobileno}" "[${time_str}@${ip_addr}]Warning: ${proc_name} is not runing"
	}
	else
    {
        echo "[${time_str}]Info: ${proc_name} is runing"
    }
    fi
}

static_by_field()
{
    typeset sep=$1
    typeset field=$2
    typeset num=$3
    
    awk -F "$sep" '{ 
       if(sum[$'"$field"']=="") { sum[$'"$field"']=0 }
       sum[$'"$field"']+='"$num"';
    }
    END {
       for (item in sum)
       {
           print item " "sum[item]
       }
    }'
}