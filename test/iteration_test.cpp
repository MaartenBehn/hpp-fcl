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
  GJK gjk_nesterov(max_iterations, tolerance);
  gjk_nesterov.gjk_variant = GJKVariant::NesterovAcceleration;

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
  
  GJK::Status res_nesterov_gjk_1 =
      gjk_nesterov.evaluate(mink_diff, init_guess, init_support_guess);
  Vec3f ray_nesterov = gjk_nesterov.ray;
  
  // Make sure GJK and Nesterov accelerated GJK find the same distance between
  // the shapes
  BOOST_CHECK(res_nesterov_gjk_1 == res_gjk_1);
  BOOST_CHECK_SMALL(fabs(ray_gjk.norm() - ray_nesterov.norm()), 1e-4);

  // Make sure GJK and Nesterov accelerated GJK converges in a reasonable
  // amount of iterations
  BOOST_CHECK(gjk.getIterations() < max_iterations);
  BOOST_CHECK(gjk_nesterov.getIterations() < max_iterations);
  std::cout << "gjk iterations:\n" << gjk.getIterations() << "\n";
  std::cout << "gjk nesterov iterations:\n" << gjk_nesterov.getIterations() << "\n";

  std::cout << "gjk dist:\n" << gjk_nesterov.distance << "\n";
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

/*
{
    "case": 3,
    "collider1": {
        "typ": "Capsule",
        "collider2origin": [
            [
                -0.758961,
                0.639032,
                0.124964,
                0.068672
            ],
            [
                -0.237632,
                -0.450513,
                0.860563,
                0.111004
            ],
            [
                0.606226,
                0.623438,
                0.493776,
                1.103138
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0
            ]
        ],
        "radius": 0.357657,
        "height": 0.472496
    },
    "collider2": {
        "typ": "Capsule",
        "collider2origin": [
            [
                0.362513,
                0.093807,
                -0.927246,
                -0.868677
            ],
            [
                -0.664387,
                0.723738,
                -0.186528,
                -1.524819
            ],
            [
                0.653585,
                0.683669,
                0.324689,
                -0.989982
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0
            ]
        ],
        "radius": 0.717198,
        "height": 0.047953
    },
    "distance": 1.5253071099939222
},
*/
