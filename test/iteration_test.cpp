/*
 *  Software License Agreement (BSD License)
 *
 *  Copyright (c) 2022, INRIA
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

/** \author Louis Montaut */

#define BOOST_TEST_MODULE FCL_NESTEROV_GJK
#include <boost/test/included/unit_test.hpp>

#include <Eigen/Geometry>
#include <hpp/fcl/narrowphase/narrowphase.h>
#include <hpp/fcl/shape/geometric_shapes.h>
#include <hpp/fcl/internal/tools.h>

#include "utility.h"

using hpp::fcl::Box;
using hpp::fcl::Capsule;
using hpp::fcl::Sphere;
using hpp::fcl::Cylinder;
using hpp::fcl::constructPolytopeFromEllipsoid;
using hpp::fcl::Convex;
using hpp::fcl::Ellipsoid;
using hpp::fcl::FCL_REAL;
using hpp::fcl::GJKSolver;
using hpp::fcl::GJKVariant;
using hpp::fcl::ShapeBase;
using hpp::fcl::support_func_guess_t;
using hpp::fcl::Transform3f;
using hpp::fcl::Triangle;
using hpp::fcl::Vec3f;
using hpp::fcl::details::GJK;
using hpp::fcl::details::MinkowskiDiff;
using std::size_t;


void test_nesterov_gjk(const ShapeBase& shape0, const ShapeBase& shape1, Transform3f& transform0, Transform3f& transform1) {
  // Solvers
  unsigned int max_iterations = 128;
  FCL_REAL tolerance = 1e-6;
  GJK gjk(max_iterations, tolerance);
  gjk.gjk_variant = GJKVariant::NesterovAcceleration;

  // Minkowski difference
  MinkowskiDiff mink_diff;

  // Same init for both solvers
  Vec3f init_guess = Vec3f(1, 0, 0);
  support_func_guess_t init_support_guess;
  init_support_guess.setZero();

  mink_diff.set(&shape0, &shape1, transform0, transform1);

  // Evaluate both solvers twice, make sure they give the same solution
  GJK::Status res_gjk_1 =
      gjk.evaluate(mink_diff, init_guess, init_support_guess);
  Vec3f ray_gjk = gjk.ray;
  
  
  // Make sure GJK and Nesterov accelerated GJK find the same distance between
  // the shapes
  // BOOST_CHECK(res_nesterov_gjk_1 == res_gjk_1);
  // BOOST_CHECK_SMALL(fabs(ray_gjk.norm() - ray_nesterov.norm()), 1e-4);

  // Make sure GJK and Nesterov accelerated GJK converges in a reasonable
  // amount of iterations
  BOOST_CHECK(gjk.getIterations() < max_iterations);
  std::cout << "gjk iterations:\n" << gjk.getIterations() << "\n";

  std::cout << "gjk dist:\n" << gjk.distance << "\n";
}

BOOST_AUTO_TEST_CASE(case_3) {
  Capsule capsule0 = Capsule(0.357657, 0.472496);
  Capsule capsule1 = Capsule(0.717198, 0.047953);

  Transform3f transform0; 
  Eigen::Matrix3d m0;
  m0 << -0.758961, 0.639032, 0.124964, 
    -0.237632, -0.450513, 0.860563,
    0.606226, 0.623438, 0.493776; 
  transform0.setTransform(m0, Vec3f(0.068672, 0.111004, 1.103138));

  Transform3f transform1; 
  Eigen::Matrix3d m1;
  m1 << 0.362513, 0.093807, -0.927246,
    -0.664387, 0.723738, -0.186528,
    0.653585, 0.683669, 0.324689;
  transform1.setTransform(m1, Vec3f(-0.868677, -1.524819, -0.989982));

  test_nesterov_gjk(capsule0, capsule1, transform0, transform1);
}


BOOST_AUTO_TEST_CASE(sphere0) {
  Sphere sphere0 = Sphere(1.0);

  Transform3f transform; 
  Eigen::Matrix3d m;
  m << 1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.0, 0.0, 1.0;
  transform.setTransform(m, Vec3f(0.0, 0.0, 0.0));

  test_nesterov_gjk(sphere0, sphere0, transform, transform);
}


BOOST_AUTO_TEST_CASE(sphere1) {
  Sphere sphere0 = Sphere(1.0);

  Transform3f transform0; 
  Transform3f transform1; 
  Eigen::Matrix3d m;
  m << 1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.0, 0.0, 1.0;
  transform0.setTransform(m, Vec3f(0.0, 0.0, 0.0));
  transform1.setTransform(m, Vec3f(1.0, 1.0, 1.0));

  test_nesterov_gjk(sphere0, sphere0, transform0, transform1);
}



BOOST_AUTO_TEST_CASE(cylinder0) {
  Cylinder cylinder0 = Cylinder(1.0, 1.0);

  Transform3f transform0; 
  Transform3f transform1; 
  Eigen::Matrix3d m;
  m << 1.0, 0.0, 0.0,
       0.0, 1.0, 0.0,
       0.0, 0.0, 1.0;
  transform0.setTransform(m, Vec3f(0.0, 0.0, 0.0));
  transform1.setTransform(m, Vec3f(3.0, 0.0, 0.0));

  test_nesterov_gjk(cylinder0, cylinder0, transform0, transform0);
}


BOOST_AUTO_TEST_CASE(box_ellipsoid) {
  std::cout << "box vs ellipsoid\n";

  Box box = Box(0.9851687611236369, 0.6982098343323029, 0.677257878780069);
  Ellipsoid ellip = Ellipsoid(0.3593240683640023, 0.88694036511957, 0.45439896964319315);

  Transform3f transform0; 
  Eigen::Matrix3d m0;
  m0 << 0.4808796989840245,-0.3349615224029876,-0.8102811201148072,
        0.8360194692307739,0.4536904964984152,0.30860392164409806,
        0.2642464042747396,-0.8258121529116014,0.49820490356014824;
  transform0.setTransform(m0, Vec3f(0.6002339404616878, 0.3190562453526661, -0.5854086434637558));

  Transform3f transform1; 
  Eigen::Matrix3d m1;
  m1 << 0.14585349627731214,-0.95228226971383,-0.26811422269689517,
        0.9565612709936813,0.06659895367350475,0.28382232858700096,
        -0.25242284456569547,-0.2978641605804072,0.9206300285039002;
  
  transform1.setTransform(m1, Vec3f(1.1424349621578582, -0.6957915756145912, -0.43826228780375104));

  test_nesterov_gjk(box, ellip, transform0, transform1);
}


BOOST_AUTO_TEST_CASE(cylinder_capsule) {
  std::cout << "cylinder vs capluse\n";

  Cylinder cylinder = Cylinder(0.9517288501403893, 0.805970533850143);
  Capsule capsule = Capsule(0.2561361758719648, 0.5731060579489073);

  Transform3f transform0; 
  Eigen::Matrix3d m0;
  m0 << -0.25626833324319087,0.25603744156860286,0.9320790577476575,
        -0.030626673490320333,0.961645599043082,-0.27257980246524155,
        -0.9661203590897328,-0.09840005262232863,-0.23859773971320178;
  transform0.setTransform(m0, Vec3f(0.48494859398701334, 0.7792352218814497, 0.3698723560514728));

  Transform3f transform1; 
  Eigen::Matrix3d m1;
  m1 << 0.9841674109221822,0.03512555425047435,-0.17372594140578287,
        0.12663457717285598,-0.8251358513985908,0.5505583625747361,
        -0.1240088349392111,-0.5638413093906118,-0.8165199242404926;
  transform1.setTransform(m1, Vec3f(-0.10338487151829642, 0.25488091367873306, -0.24370987750835632));

  test_nesterov_gjk(cylinder, capsule, transform0, transform1);
}


BOOST_AUTO_TEST_CASE(capluse_sphere) {
  std::cout << "capluse vs sphere\n";

  Capsule capsule = Capsule(0.9218123729576948, 0.16552090305800327);
  Sphere sphere = Sphere(0.34425784144283245);

  Transform3f transform0; 
  Eigen::Matrix3d m0;
  m0 << 0.27841557, -0.58984435, -0.75800291, 
        0.91747359, -0.07013484,  0.39156521, 
       -0.28412494, -0.8044655,   0.52163998;
  transform0.setTransform(m0, Vec3f(0.48459252, -1.33979643, 1.01785838));

  Transform3f transform1; 
  Eigen::Matrix3d m1;
  m1 << 1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0;
  transform1.setTransform(m1, Vec3f(-1.66774998, -2.96656368, -0.05856713));

  test_nesterov_gjk(capsule, sphere, transform0, transform1);
}


BOOST_AUTO_TEST_CASE(cylinder_cylinder) {
  std::cout << "cylinder vs cylinder\n";

  Cylinder cylinder0 = Cylinder(0.510938, 0.127116);
  Cylinder cylinder1 = Cylinder(0.903175, 0.456057);
  

  Transform3f transform0; 
  Eigen::Matrix3d m0;
  m0 << 0.365685,  0.898448, -0.243034,
        -0.739601,  0.439027,  0.510143,
        0.565035, -0.006803,  0.825039;
  transform0.setTransform(m0, Vec3f(-1.556104, 0.765753, -0.834356));

  Transform3f transform1; 
  Eigen::Matrix3d m1;
  m1 << 0.946613,  0.007576,  0.322282,
        0.285324,  0.44562,  -0.848536,
       -0.150044,  0.89519,   0.419668;
  transform1.setTransform(m1, Vec3f(0.394209, 0.433601, 0.332314));

  test_nesterov_gjk(cylinder0, cylinder1, transform0, transform1);
}
