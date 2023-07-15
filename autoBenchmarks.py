from asyncio.subprocess import PIPE
import subprocess
import os
from SendSMS import sendSMS

from numpy import ones

cktToken="e077fe62-2130-11ee-ad8c-0c9d92c3ff60"
cktDir="/home/templarares/acme/"+cktToken+"/checkpoints/d4pg_learner"
learnerDir="/home/templarares/acmeNew/lib/python3.7/site-packages/acme/agents/tf/d4pg/"
if os.path.exists("autoBenchmarksResult"):
    os.remove("autoBenchmarksResult")
#while (True):e
rewards=ones(24)
for i in range(24):
    #remove becnhmark file
    if os.path.exists("BenchmarkResultTrue.txt"):
        os.remove("BenchmarkResultTrue.txt")
    #run benchmark once
    #p=subprocess.call(["/home/templarares/acmeNew/bin/python","acmeIngress_benchmark.py"])
    p=subprocess.Popen(["/home/templarares/acmeNew/bin/python","acmeIngress_benchmark.py"],stdin=PIPE,stdout=PIPE)
    p.wait()
    out ,err = p.communicate()
    avgIndex=out.index(b"average reward")
    eol=out.index(b"\n",avgIndex)
    avgReward=float(out[avgIndex+18:eol])
    with open("autoBenchmarksResult",'a') as file:
        file.write("average reward of run %d is:%d\n"%(i,avgReward))
        file.close()
    rewards[i]=avgReward
    #regress checkpoints by 1 commit
    p=subprocess.Popen(['git','log'],cwd=cktDir,stdin=PIPE,stdout=PIPE)
    out ,err = p.communicate(input=b"q")
    idx=out.index(b"commit",20)
    commit=out[idx+7:idx+12]
    p.wait()
    p.kill()

    p=subprocess.Popen(['git','checkout',commit],cwd=cktDir)
    p.wait()
    p.kill()

    p=subprocess.Popen(['ls'],cwd=cktDir,stdin=PIPE,stdout=PIPE)
    out ,err = p.communicate()
    p.wait()
    p.kill()


    #modify the ckt reference (number) in learning.py
    fst=out.index(b"-")
    snd=out.index(b".")
    cktNumber=out[fst+1:snd]
    print(cktNumber)

    with open(learnerDir+"learning.py",'r') as input:
        with open("temp.py",'w') as output: 
            for line in input:
                if cktToken not in line.strip("\n"):
                    output.write(line)
                else:
                    output.write("      ckpt = \'/home/templarares/acme/"+cktToken+"/checkpoints/d4pg_learner/ckpt-"+cktNumber.decode("utf-8")+b"\'\n".decode("utf-8"))\

    os.replace('temp.py',learnerDir+"learning.py")
print(rewards)
sendSMS()