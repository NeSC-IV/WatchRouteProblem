#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Simple_polygon_visibility_2.h>
#include <CGAL/Triangular_expansion_visibility_2.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/Arr_segment_traits_2.h>
#include <CGAL/Arr_naive_point_location.h>
#include <istream>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <utility>
#include <exception>
typedef CGAL::Exact_predicates_exact_constructions_kernel               Kernel;
typedef Kernel::Point_2                                                 Point_2;
typedef Kernel::Segment_2                                               Segment_2;
typedef CGAL::Arr_segment_traits_2<Kernel>                              Traits_2;
typedef CGAL::Arrangement_2<Traits_2>                                   Arrangement_2;
typedef CGAL::Triangular_expansion_visibility_2<Arrangement_2>  TEV;

  std::vector<std::pair<double, double> > compute_visibility_cpp(std::vector<std::pair<double,double>> pointList, std::pair<double,double> watcher) {
    // int main(){
      //create environment
      std::vector<Point_2> points;
      std::vector<Segment_2> segments;
      std::vector<std::pair<double,double>> result;
      for (auto point = pointList.begin();point!=pointList.end();++point){
        Point_2 p1(point->first,point->second);
        points.push_back(p1);
      }
      int n = points.size();
      for (int i=0;i<n;++i){
        segments.push_back(Segment_2(points[i], points[(i+1)%n]));
      }

      Arrangement_2 env;
      CGAL::insert_non_intersecting_curves(env,segments.begin(),segments.end());
      // find the face of the query point
      Point_2 q(watcher.first, watcher.second);
      Arrangement_2::Face_const_handle * face;
      CGAL::Arr_naive_point_location<Arrangement_2> pl(env);
      CGAL::Arr_point_location_result<Arrangement_2>::Type obj = pl.locate(q);
      // The query point locates in the interior of a face
      face = boost::get<Arrangement_2::Face_const_handle> (&obj);
      // compute non regularized visibility area
      // typedef CGAL::Simple_polygon_visibility_2<Arrangement_2, CGAL::Tag_true> NSPV;
      Arrangement_2 non_regular_output;
      // NSPV non_regular_visibility(env);
      typedef CGAL::Triangular_expansion_visibility_2<Arrangement_2>  TEV;
      TEV non_regular_visibility(env);
      non_regular_visibility.compute_visibility(q, *face, non_regular_output);
      double x,y;
      for (auto eit = non_regular_output.vertices_begin(); eit != non_regular_output.vertices_end(); ++eit){
        x = CGAL::to_double(eit->point().x());
        y = CGAL::to_double(eit->point().y());
        result.push_back(std::make_pair(x,y));
      }
      return result;
  }


  PYBIND11_MODULE(visibility, m) {
    m.def("compute_visibility_cpp", &compute_visibility_cpp, "A function that adds two numbers");
  }
