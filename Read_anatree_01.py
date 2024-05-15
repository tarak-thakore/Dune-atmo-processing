from array import array
from ROOT import TFile, TTree

f1 = TFile.Open('ana_tree_hd_9993.root')
print(type(f1))
t1 = f1.Get('analysistree/anatree')
print(type(t1))
#t1.Show(0)

'''
leaves = t1.GetListOfLeaves()
for l in leaves:
  print(l))
'''

branches = ['enu_truth','nuPDG_truth','nuvtxx_truth','nuvtxy_truth','nuvtxz_truth']


#pyroot approach - replica of how a TTree is read in C++ ROOT - inefficient
n = array('f', [ 0. ])
t1.SetBranchAddress('enu_truth',n)
'''
for b in branches:
  #t1.SetBranchAddress(b)
  print(b)

print("Total entries:",t1.GetEntries())
for iev in range(5):
  t1.GetEntry(iev)
  print(n)
'''

for iev in t1:
  print(iev.enu_truth)

