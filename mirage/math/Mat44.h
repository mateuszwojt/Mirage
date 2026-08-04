#pragma once

#include "mirage/math/Vec4.h"
#include "mirage/math/Transform.h"

namespace Mirage
{
	struct Mat44
	{
		CUDA_CALLABLE inline Mat44() { std::memset(this, 0, sizeof(*this)); }

		CUDA_CALLABLE inline Mat44(Real m11, Real m12, Real m13, Real m14,
								   Real m21, Real m22, Real m23, Real m24,
								   Real m31, Real m32, Real m33, Real m34,
								   Real m41, Real m42, Real m43, Real m44)
		{
			// col 1
			cols[0][0] = m11;
			cols[0][1] = m21;
			cols[0][2] = m31;
			cols[0][3] = m41;

			// col 2
			cols[1][0] = m12;
			cols[1][1] = m22;
			cols[1][2] = m32;
			cols[1][3] = m42;

			// col 3
			cols[2][0] = m13;
			cols[2][1] = m23;
			cols[2][2] = m33;
			cols[2][3] = m43;

			// col 4
			cols[3][0] = m14;
			cols[3][1] = m24;
			cols[3][2] = m34;
			cols[3][3] = m44;
		}

		CUDA_CALLABLE inline Mat44(const Vec4 &c1, const Vec4 &c2, const Vec4 &c3, const Vec4 &c4)
		{
			SetCol(0, c1);
			SetCol(1, c2);
			SetCol(2, c3);
			SetCol(3, c4);
		}

		CUDA_CALLABLE inline Mat44(const Transform &t)
		{
			Mat33 r(t.r);

			SetCol(0, Vec4(r.GetCol(0) * t.s, 0.0));
			SetCol(1, Vec4(r.GetCol(1) * t.s, 0.0));
			SetCol(2, Vec4(r.GetCol(2) * t.s, 0.0));
			SetCol(3, Vec4(t.p * t.s, 1.0));
		}

		CUDA_CALLABLE inline static Mat44 Identity()
		{
			return Mat44(1.0f, 0.0f, 0.0f, 0.0f,
						 0.0f, 1.0f, 0.0f, 0.0f,
						 0.0f, 0.0f, 1.0f, 0.0f,
						 0.0f, 0.0f, 0.0f, 1.0f);
		}

		CUDA_CALLABLE inline Vec4 GetCol(int index) const { return Vec4(cols[index][0], cols[index][1], cols[index][2], cols[index][3]); }
		CUDA_CALLABLE inline void SetCol(int index, const Vec4 &v)
		{
			cols[index][0] = v.x;
			cols[index][1] = v.y;
			cols[index][2] = v.z;
			cols[index][3] = v.w;
		}

		Real cols[4][4]; // column major format
	};

	CUDA_CALLABLE inline Vec4 Multiply(const Mat44 &a, const Vec4 &v)
	{
		Vec4 result;
		::Mirage::MatrixMultiply<4, 4, 1>(&result.x, &a.cols[0][0], &v.x);
		return result;
	}

	CUDA_CALLABLE inline Mat44 Multiply(const Mat44 &a, const Mat44 &b)
	{
		Mat44 result;
		::Mirage::MatrixMultiply<4, 4, 4>(&result.cols[0][0], &a.cols[0][0], &b.cols[0][0]);
		return result;
	}

	CUDA_CALLABLE inline Vec4 operator*(const Mat44 &a, const Vec4 &v) { return Multiply(a, v); }
	CUDA_CALLABLE inline Mat44 operator*(const Real s, const Mat44 &a)
	{
		Mat44 result;
		::Mirage::MatrixScale<4, 4>(&result.cols[0][0], &a.cols[0][0], s);
		return result;
	}

	CUDA_CALLABLE inline Mat44 operator*(const Mat44 &a, const Real s) { return s * a; }
	CUDA_CALLABLE inline Mat44 operator*(const Mat44 &a, const Mat44 &b) { return Multiply(a, b); }

	CUDA_CALLABLE inline Mat44 operator-(const Mat44 &a, const Mat44 &b)
	{
		Mat44 result;
		::Mirage::MatrixSub<4, 4>(&result.cols[0][0], &a.cols[0][0], &b.cols[0][0]);
		return result;
	}

	CUDA_CALLABLE inline Mat44 operator+(const Mat44 &a, const Mat44 &b)
	{
		Mat44 result;
		::Mirage::MatrixAdd<4, 4>(&result.cols[0][0], &a.cols[0][0], &b.cols[0][0]);
		return result;
	}

	CUDA_CALLABLE inline Mat44 &operator+=(Mat44 &a, const Mat44 &b) { return a = a + b; }
	CUDA_CALLABLE inline Mat44 &operator-=(Mat44 &a, const Mat44 &b) { return a = a - b; }

	CUDA_CALLABLE inline Vec3 TransformVector(const Mat44 &a, const Vec3 &v)
	{
		Vec3 result;
		result.x = a.cols[0][0] * v.x + a.cols[1][0] * v.y + a.cols[2][0] * v.z;
		result.y = a.cols[0][1] * v.x + a.cols[1][1] * v.y + a.cols[2][1] * v.z;
		result.z = a.cols[0][2] * v.x + a.cols[1][2] * v.y + a.cols[2][2] * v.z;
		return result;
	}

	CUDA_CALLABLE inline Vec3 TransformPoint(const Mat44 &a, const Vec3 &v)
	{
		Vec3 result;
		result.x = a.cols[0][0] * v.x + a.cols[1][0] * v.y + a.cols[2][0] * v.z + a.cols[3][0];
		result.y = a.cols[0][1] * v.x + a.cols[1][1] * v.y + a.cols[2][1] * v.z + a.cols[3][1];
		result.z = a.cols[0][2] * v.x + a.cols[1][2] * v.y + a.cols[2][2] * v.z + a.cols[3][2];
		return result;
	}

	CUDA_CALLABLE inline Mat44 Transpose(const Mat44 &a)
	{
		Mat44 t;
		::Mirage::MatrixTranspose<4, 4>(&t.cols[0][0], &a.cols[0][0]);
		return t;
	}

	CUDA_CALLABLE inline Mat44 AffineInverse(const Mat44 &m)
	{
		Mat44 inv;

		// transpose upper 3x3
		for (int c = 0; c < 3; ++c)
		{
			for (int r = 0; r < 3; ++r)
			{
				inv.cols[c][r] = m.cols[r][c];
			}
		}

		// multiply -translation by upper 3x3 transpose
		inv.cols[3][0] = -Dot(m.GetCol(3), m.GetCol(0));
		inv.cols[3][1] = -Dot(m.GetCol(3), m.GetCol(1));
		inv.cols[3][2] = -Dot(m.GetCol(3), m.GetCol(2));
		inv.cols[3][3] = 1.0f;

		return inv;
	}

	CUDA_CALLABLE inline Mat44 LookAtMatrix(const Vec3 &viewer, const Vec3 &target)
	{
		// create a basis from viewer to target (OpenGL convention looking down -z)
		Vec3 forward = -Normalize(target - viewer);
		Vec3 up(0.0f, 1.0f, 0.0f);
		Vec3 left = -Normalize(Cross(forward, up));
		up = -Cross(left, forward);

		Mat44 xform(left.x, up.x, forward.x, viewer.x,
					left.y, up.y, forward.y, viewer.y,
					left.z, up.z, forward.z, viewer.z,
					0.0f, 0.0f, 0.0f, 1.0f);

		return AffineInverse(xform);
	}

	// generate a rotation matrix around an axis, from PBRT p74
	CUDA_CALLABLE inline Mat44 RotationMatrix(float angle, const Vec3 &axis)
	{
		Vec3 a = Normalize(axis);
		float s = sinf(angle);
		float c = cosf(angle);

		float m[4][4];

		m[0][0] = a.x * a.x + (1.0f - a.x * a.x) * c;
		m[0][1] = a.x * a.y * (1.0f - c) + a.z * s;
		m[0][2] = a.x * a.z * (1.0f - c) - a.y * s;
		m[0][3] = 0.0f;

		m[1][0] = a.x * a.y * (1.0f - c) - a.z * s;
		m[1][1] = a.y * a.y + (1.0f - a.y * a.y) * c;
		m[1][2] = a.y * a.z * (1.0f - c) + a.x * s;
		m[1][3] = 0.0f;

		m[2][0] = a.x * a.z * (1.0f - c) + a.y * s;
		m[2][1] = a.y * a.z * (1.0f - c) - a.x * s;
		m[2][2] = a.z * a.z + (1.0f - a.z * a.z) * c;
		m[2][3] = 0.0f;

		m[3][0] = 0.0f;
		m[3][1] = 0.0f;
		m[3][2] = 0.0f;
		m[3][3] = 1.0f;

		return (Mat44 &)m;
	}

	CUDA_CALLABLE inline Mat44 TranslationMatrix(const Vec3 &t)
	{
		Mat44 m = Mat44::Identity();
		m.SetCol(3, Vec4(t, 1.0f));
		return m;
	}

	CUDA_CALLABLE inline Mat44 ScaleMatrix(const Vec3 &s)
	{
		float m[4][4] = {{s.x, 0.0f, 0.0f, 0.0f},
						 {0.0f, s.y, 0.0f, 0.0f},
						 {0.0f, 0.0f, s.z, 0.0f},
						 {0.0f, 0.0f, 0.0f, 1.0f}};

		return (Mat44 &)m;
	}

	CUDA_CALLABLE inline Mat44 OrthographicMatrix(float left, float right, float bottom, float top, float n, float f)
	{

		float m[4][4] = {{2.0f / (right - left), 0.0f, 0.0f, 0.0f},
						 {0.0f, 2.0f / (top - bottom), 0.0f, 0.0f},
						 {0.0f, 0.0f, -2.0f / (f - n), 0.0f},
						 {-(right + left) / (right - left), -(top + bottom) / (top - bottom), -(f + n) / (f - n), 1.0f}};

		return (Mat44 &)m;
	}

	// this is designed as a drop in replacement for gluPerspective
	CUDA_CALLABLE inline Mat44 ProjectionMatrix(float fov, float aspect, float znear, float zfar)
	{
		float f = 1.0f / tanf(::Mirage::DegToRad(fov * 0.5f));
		float zd = znear - zfar;

		float view[4][4] = {{f / aspect, 0.0f, 0.0f, 0.0f},
							{0.0f, f, 0.0f, 0.0f},
							{0.0f, 0.0f, (zfar + znear) / zd, -1.0f},
							{0.0f, 0.0f, (2.0f * znear * zfar) / zd, 0.0f}};

		return (Mat44 &)view;
	}

	CUDA_CALLABLE inline Mat44 InterpolateTransform(const Mat44 &a, const Mat44 &b, float time)
	{
		// todo: rotation interpolation using quaternions
		Mat44 c = b;
		c.SetCol(3, ::Mirage::Lerp(a.GetCol(3), b.GetCol(3), time));

		return c;
	}
} // namespace Mirage
