from SubstrateNetworkManager import SubstrateNetwork as sn
from VirtualNetworkManager import VirtualNodes as vn
from AllocationManager import PeSA
from RequestManager import GenerateRequest as greq
sn = sn()
vn = vn()
PeSA = PeSA()
greq = greq()
print(vn.Gv.nodes(data=True))