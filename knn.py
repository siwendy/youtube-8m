#!/usr/bin/python
import sys
import math

vids = []
vid_map = {}
for line in open(sys.argv[1]):
  vid,feas = line.strip().split('\t')
  f = [float(x) for x in feas.split(' ')]
  sqrt = math.pow(sum([x*x for x in f]), 0.5)
  vid_map[vid] = len(vids)
  vids.append([vid, f, sqrt])
  if len(vids) % 10000 == 0:
    print(len(vids),'done')
  #if len(vids) > 1000:
  #  break

print('vids num:{}'.format(len(vids)), file=sys.stderr)

while True:
  vid = sys.stdin.readline().strip()
  if vid not in vid_map:
    print('not find vid :{}'.format(vid), file=sys.stderr)
    continue
  idx = vid_map[vid]
  scores = []
  for can in vids:
    s = sum([x*y for x,y in zip(vids[idx][1], can[1])])
    s_cos =  s/(can[2]*vids[idx][2])
    scores.append([s, s_cos, can[0]])
  scores1= sorted(scores, key=lambda x:x[0], reverse=True)
  scores2= sorted(scores, key=lambda x:x[1], reverse=True)
  for s1,s2 in zip(scores1[:20],scores2[:20]):
    print('{}\t{}\t==\t{}\t{}'.format(s1[2], s1[0], s2[2], s2[1]))
