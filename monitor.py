import psutil
import sys, getopt
from SendSMS import sendSMS
import time
import signal
import os
import subprocess

# #use twilio SMS to notify auto-termination
# from twilio.rest import Client
# # Your Account SID from twilio.com/console
# account_sid = "ACd910fbbc47da001bd0fedcb5244cde07"
# # Your Auth Token from twilio.com/console
# auth_token  = "c0c0d3a6a1340e70aa1e89d9808adc9f"
# client = Client(account_sid, auth_token)
# message = client.messages.create(
#     # 这里中国的号码前面需要加86
#     to="+8613809008267", 
#     from_="+14356592468",
#     body="This is a notification message for program termination")
# print(message.sid)

# #use IFTTT to send notification
# import requests
 
# def send_notice(event_name, key, text):
#     url = "https://maker.ifttt.com/trigger/"+event_name+"/with/key/"+key+""
#     payload = "{\"value1\": \""+text+"\"}"
#     headers = {
#     'Content-Type': "application/json",
#     'User-Agent': "PostmanRuntime/7.15.0",
#     'Accept': "*/*",
#     'Cache-Control': "no-cache",
#     'Postman-Token': "a9477d0f-08ee-4960-b6f8-9fd85dc0d5cc,d376ec80-54e1-450a-8215-952ea91b01dd",
#     'Host': "maker.ifttt.com",
#     'accept-encoding': "gzip, deflate",
#     'content-length': "63",
#     'Connection': "keep-alive",
#     'cache-control': "no-cache"
#     }
 
#     response = requests.request("POST", url, data=payload.encode('utf-8'), headers=headers)
 
#     print(response.text)
 
# text = "Training complete!"
# send_notice('notify_phone', 'ciupL2jYLeb8biRCYTfUnr', text)

def terminate_acme():
    PROCNAME="acmeIngress_mp.py"
    sig=signal.SIGTERM
    for proc in psutil.process_iter():
        for cmd in proc.cmdline():
            if PROCNAME in cmd :
                #print(proc.cmdline())
                #proc.kill()
                os.kill(proc.pid,sig)
                break
if __name__=='__main__':
    #args=sys.argv
    #token=args[1]
    token="acmeSessionID"
    cktToken="checkoutpointToken"
    length=10
    totalReward=0.0
    avgReward=0.0
    NLoops=0
    RewardThreshold=70000
    NLoopsThreshold=10000
                        
    opts,args=getopt.getopt(sys.argv[1:],"ht:l:r:n:c:")
    for opt,arg in opts:
        if opt=="-h":
            print ('monitory.py \n-t <acme session token> \n-l <length for averging evaluator reward > \
                \n-r <reward threshold for auto termination> \n-n <NLoops threshold for auto termination>')
            sys.exit()
        elif opt=="-t":
            token=arg
        elif opt=="-c":
            cktToken=arg
        elif opt=="-l":
            length=int(arg)
        elif opt=="-r":
            RewardThreshold=int(arg)
        elif opt=="-n":
            NLoopsThreshold=int(arg)
    filename = "/home/templarares/acme/"+token+"/logs/evaluator/logs.csv"
    filename2="/home/templarares/devel/src/bit-car-inout-controller/etc/NLoops.yaml"
    cktDir="/home/templarares/acme/"+cktToken+"/checkpoints/d4pg_learner"
    while True:
        with open(filename, 'rb') as fp:
            line_offset=0
            offset=-80*(length+line_offset)
            fp.seek(offset,2)
            lines=fp.readlines()
            #calculate reward for the latest 10 runs
            
            for i in range(line_offset+1,line_offset+length+1):
                line=lines[-i].decode()
                #print(line)
                fst=line.index(",")
                #print("first is %d"%fst)
                snd=line.index(",",fst+1)
                #print("second is %d"%snd)
                reward=float(line[fst+1:snd])
                
                #second version, with two more commas
                trd=line.index(",",snd+1)
                fth=line.index(",",trd+1)
                reward=float(line[trd+1:fth])
                #print(reward)
                totalReward+=reward
            avgReward=totalReward/length
            print(avgReward)
            fp.close()
        with open(filename2, 'r') as fp:
            first_line=fp.readline().strip('\n')
            comma=first_line.find(":")
            if comma>0:
                NLoops=int(first_line[comma+2:])
            else:
                NLoops=1
            print("NLoops= %d"%NLoops)
        if avgReward>RewardThreshold and NLoops>NLoopsThreshold:
            #make snapshots of the checkpoint several times, wait for the optimal networks to be saved locally
            #time.sleep(60)
            p=subprocess.Popen(['git','init'],cwd=cktDir)
            p.wait()
            p.kill()
            for i in range(10):
                p=subprocess.Popen(['git','add','.'],cwd=cktDir)
                p.wait()
                p.kill()
                commitMsg='\"snapshot%d\"'%i
                p=subprocess.Popen(['git','commit','-m',commitMsg],cwd=cktDir)
                p.wait()
                p.kill()
                time.sleep(42)
            
            sendSMS()
            terminate_acme()
            sys.exit("Training completion criteria met!")
        totalReward=0
        time.sleep(42)