#pragma once

#include "mirage/math/Vec2.h"

namespace Mirage
{
	struct Mat22
	{

		CUDA_CALLABLE inline Mat22()
		{
			m[0][0] = 0.0;
			m[0][1] = 0.0;
			m[1][0] = 0.0;
			m[1][1] = 0.0;
		}

		CUDA_CALLABLE inline Mat22(Real m11, Real m12, Real m21, Real m22)
		{
			m[0][0] = m11;
			m[0][1] = m21;
			m[1][0] = m12;
			m[1][1] = m22;
		}

		CUDA_CALLABLE inline Mat22(const Vec2 &c1, const Vec2 &c2)
		{
			SetCol(0, c1);
			SetCol(1, c2);
		}

		CUDA_CALLABLE inline static Mat22 Identity()
		{
			return Mat22(Vec2(1.0, 0.0), Vec2(0.0, 1.0));
		}

		CUDA_CALLABLE inline Vec2 GetCol(int index) const { return Vec2(m[index][0], m[index][1]); }
		CUDA_CALLABLE inline void SetCol(int index, const Vec2 &v)
		{
			m[index][0] = v.x;
			m[index][1] = v.y;
		}

		Real m[2][2];
	};

	CUDA_CALLABLE inline Vec2 Multiply(const Mat22 &a, const Vec2 &v)
	{
		Vec2 result;
		MatrixMultiply<2, 2, 1>(&result.x, &a.m[0][0], &v.x);
		return result;
	}

	CUDA_CALLABLE inline Mat22 operator+(const Mat22 &a, const Mat22 &b)
	{
		Mat22 s;
		MatrixAdd<2, 2>(&s.m[0][0], &a.m[0][0], &b.m[0][0]);
		return s;
	}

	CUDA_CALLABLE inline Mat22 operator-(const Mat22 &a, const Mat22 &b)
	{
		Mat22 s;
		MatrixSub<2, 2>(&s.m[0][0], &a.m[0][0], &b.m[0][0]);
		return s;
	}

	CUDA_CALLABLE inline Mat22 operator*(const Mat22 &a, const Mat22 &b)
	{
		Mat22 result;
		MatrixMultiply<2, 2, 2>(&result.m[0][0], &a.m[0][0], &b.m[0][0]);
		return result;
	}

	// matrix multiplcation
	CUDA_CALLABLE inline Vec2 operator*(const Mat22 &a, const Vec2 &v) { return Multiply(a, v); }

	// scalar multiplication
	CUDA_CALLABLE inline Mat22 operator*(const Mat22 &a, Real s)
	{
		Mat22 result;
		MatrixScale<2, 2>(&result.m[0][0], &a.m[0][0], s);
		return result;
	}

	CUDA_CALLABLE inline Mat22 operator*(Real s, const Mat22 &a) { return a * s; }

	// unary negation
	CUDA_CALLABLE inline Mat22 operator-(const Mat22 &a) { return -1.0f * a; }

	// generate a counter clockwise rotation by theta radians
	CUDA_CALLABLE inline Mat22 RotationMatrix(Real theta)
	{
		Real cosTheta = cos(theta);
		Real sinTheta = sin(theta);

		Real m[2][2] =
			{
				{cosTheta, sinTheta},
				{-sinTheta, cosTheta}};

		return *(Mat22 *)m;
	}

	// derivative of a rotation matrix w.r.t. theta
	CUDA_CALLABLE inline Mat22 RotationMatrixDerivative(Real theta)
	{
		Real cosTheta = cos(theta);
		Real sinTheta = sin(theta);

		Real m[2][2] =
			{
				{-sinTheta, cosTheta},
				{-cosTheta, -sinTheta}};

		return *(Mat22 *)m;
	}

	CUDA_CALLABLE inline Real Determinant(const Mat22 &a)
	{
		return a.m[0][0] * a.m[1][1] - a.m[1][0] * a.m[0][1];
	}

	CUDA_CALLABLE inline Real Trace(const Mat22 &a)
	{
		return a.m[0][0] + a.m[1][1];
	}

	CUDA_CALLABLE inline Mat22 Outer(const Vec2 &a, const Vec2 &b)
	{
		return Mat22(a * b.x, a * b.y);
	}

	CUDA_CALLABLE inline Mat22 Inverse(const Mat22 &a, Real *det)
	{
		*det = Determinant(a);
		if (*det != 0.0f)
		{
			return (1.0 / (*det)) * Mat22(a.m[1][1], -a.m[1][0], -a.m[0][1], a.m[0][0]);
		}
		else
			return a;
	}

	CUDA_CALLABLE inline Mat22 Transpose(const Mat22 &a)
	{
		Mat22 t;
		MatrixTranspose<2, 2>(&t.m[0][0], &a.m[0][0]);
		return t;
	}
}