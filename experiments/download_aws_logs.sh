ips=("3.101.102.130"  "3.101.132.46" "13.56.77.16" "13.57.15.239" "13.57.192.253" "18.144.44.58" "18.144.50.110" "54.67.23.222" "54.67.68.16" "54.153.61.76" "54.177.31.216" "54.183.139.173" "54.183.159.208" "54.215.218.168" "54.241.225.157")

for ip in "${ips[@]}";
do
  echo ${ip};
  echo "scp -r -i ~/Desktop/cython_debugging.pem ubuntu@${ip}:~/BanditPAM_plusplus_experiments/experiments/logs/* logs";
  scp -r -i ~/Desktop/cython_debugging.pem ubuntu@"${ip}":~/BanditPAM_plusplus_experiments/experiments/logs/* logs
done