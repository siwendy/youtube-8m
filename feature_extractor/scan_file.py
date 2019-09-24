#!/usr/env/bin python
import os
import sys

def get_vid(filepath):
  return filepath.strip().split('/')[-1].split('.')[0]

def scan_files(path, allfile, dup_dict):
  filelist = os.listdir(path)
  for filename in filelist:
    filepath = os.path.join(path,filename)
    if os.path.isdir(filepath):
      scan_files(filepath, allfile, dup_dict)
    else:
      vid = get_vid(filepath)
      if vid not in dup_dict:
        allfile.append(filepath)
        dup_dict[vid] = 1


input_file=sys.argv[1]
dup_file=sys.argv[2]
output_file=sys.argv[3]

vid_dup_dict = {}
for line in open(dup_file):
  vid = get_vid(line.strip())
  vid_dup_dict[vid] = 1

allfile = []
scan_files(input_file, allfile, vid_dup_dict)

out = open(output_file, 'w')
for f in allfile:
  out.write('{},0\n'.format(f))
out.close()
