#include<vcg/complex/complex.h>

#include<vcg/complex/algorithms/point_sampling.h>
#include<vcg/complex/algorithms/clustering.h>

using namespace vcg;
using namespace std;

class MyEdge;
class MyFace;
class MyVertex;
struct MyUsedTypes : public UsedTypes<	Use<MyVertex>   ::AsVertexType,
                                        Use<MyEdge>     ::AsEdgeType,
                                        Use<MyFace>     ::AsFaceType>{};

class MyVertex  : public Vertex<MyUsedTypes, vertex::Coord3f, vertex::Normal3f, vertex::BitFlags  >{};
class MyFace    : public Face< MyUsedTypes, face::FFAdj,  face::Normal3f, face::VertexRef, face::BitFlags > {};
class MyEdge    : public Edge<MyUsedTypes>{};
class MyMesh    : public tri::TriMesh< vector<MyVertex>, vector<MyFace> , vector<MyEdge>  > {};

void poissonDiskSampling(MyMesh &m, MyMesh &subM, float radius)
{
  tri::MeshSampler<MyMesh> mps(subM);

  tri::UpdateBounding<MyMesh>::Box(m);

  tri::SurfaceSampling<MyMesh,tri::TrivialSampler<MyMesh> >::SamplingRandomGenerator().initialize(time(0));
  
  printf("Subsampling a PointCloud of %i vert with %f radius\n",m.VN(),radius);
  tri::SurfaceSampling<MyMesh,tri::MeshSampler<MyMesh> >::PoissonDiskParam pp;
  pp.bestSampleChoiceFlag=true;
  tri::SurfaceSampling<MyMesh,tri::MeshSampler<MyMesh> >::PoissonDiskPruning(mps, m, radius, pp);
  printf("Sampled %i vertices in %5.2f\n",subM.VN(), float(pp.pds.pruneTime+pp.pds.gridTime)/float(CLOCKS_PER_SEC));

}