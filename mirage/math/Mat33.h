#pragma once

#include "mirage/math/Vec3.h"
#include "mirage/utils/MathUtils.h"

namespace Mirage
{
	struct Mat33;
	struct Quat
	{
		CUDA_CALLABLE inline Quat() : x(0.0), y(0.0), z(0.0), w(1.0) {}
		CUDA_CALLABLE inline Quat(Real x, Real y, Real z, Real w) : x(x), y(y), z(z), w(w) {}
		CUDA_CALLABLE inline Quat(Vec3 axis, Real angle)
		{
			const Real s = sin(angle * 0.5);
			const Real c = cos(angle * 0.5);

			x = axis.x * s;
			y = axis.y * s;
			z = axis.z * s;
			w = c;
		}

		CUDA_CALLABLE inline Quat(const Mat33 &m);

		CUDA_CALLABLE inline Vec3 GetImaginary() { return Vec3(x, y, z); }

		Real x;
		Real y;
		Real z;
		Real w; // real part
	};

	CUDA_CALLABLE inline Quat operator-(const Quat &a) { return Quat(-a.x, -a.y, -a.z, -a.w); }
	CUDA_CALLABLE inline Quat operator+(const Quat &a, const Quat &b) { return Quat(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
	CUDA_CALLABLE inline Quat operator-(const Quat &a, const Quat &b) { return Quat(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w); }
	CUDA_CALLABLE inline Quat operator*(const Quat &a, Real s) { return Quat(a.x * s, a.y * s, a.z * s, a.w * s); }
	CUDA_CALLABLE inline Quat operator*(const Quat &a, const Quat &b)
	{
		return Quat(a.w * b.x + b.w * a.x + a.y * b.z - b.y * a.z,
					a.w * b.y + b.w * a.y + a.z * b.x - b.z * a.x,
					a.w * b.z + b.w * a.z + a.x * b.y - b.x * a.y,
					a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z);
	}
	CUDA_CALLABLE inline Quat operator*(Real s, const Quat &a) { return a * s; }
	CUDA_CALLABLE inline Quat operator/(const Quat &a, Real s) { return a * (1.0 / s); }

	CUDA_CALLABLE inline Quat &operator+=(Quat &a, const Quat &b) { return a = a + b; }
	CUDA_CALLABLE inline Quat &operator-=(Quat &a, const Quat &b) { return a = a - b; }
	CUDA_CALLABLE inline Quat &operator*=(Quat &a, const Quat &b) { return a = a * b; }
	CUDA_CALLABLE inline Quat &operator*=(Quat &a, Real s)
	{
		a.x *= s;
		a.y *= s;
		a.z *= s;
		a.w *= s;
		return a;
	}
	CUDA_CALLABLE inline Quat &operator/=(Quat &a, Real s)
	{
		Real rcp = 1.0 / s;
		a.x *= rcp;
		a.y *= rcp;
		a.z *= rcp;
		a.w *= rcp;
		return a;
	}

	CUDA_CALLABLE inline Quat Normalize(const Quat &q)
	{
		Real length = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
		Real rcpLength = 1.0 / length;

		return q * rcpLength;
	}

	CUDA_CALLABLE inline Quat Conjugate(const Quat &q) { return Quat(-q.x, -q.y, -q.z, q.w); }

	CUDA_CALLABLE inline Vec3 Rotate(const Quat &q, const Vec3 &v)
	{
		// q*v*q'
		Quat t = q * Quat(v.x, v.y, v.z, 0.0) * Conjugate(q);
		return t.GetImaginary();
	}

	CUDA_CALLABLE inline Vec3 operator*(const Quat &q, const Vec3 &v)
	{
		return Rotate(q, v);
	}
	struct Mat33
	{
		CUDA_CALLABLE inline Mat33() { std::memset(this, 0, sizeof(*this)); }

		CUDA_CALLABLE inline Mat33(Real m11, Real m12, Real m13,
								   Real m21, Real m22, Real m23,
								   Real m31, Real m32, Real m33)
		{
			// col 1
			m[0][0] = m11;
			m[0][1] = m21;
			m[0][2] = m31;

			// col 2
			m[1][0] = m12;
			m[1][1] = m22;
			m[1][2] = m32;

			// col 3
			m[2][0] = m13;
			m[2][1] = m23;
			m[2][2] = m33;
		}

		CUDA_CALLABLE inline Mat33(const Vec3 &c1, const Vec3 &c2, const Vec3 &c3)
		{
			SetCol(0, c1);
			SetCol(1, c2);
			SetCol(2, c3);
		}

		CUDA_CALLABLE inline Mat33(const Quat &q)
		{
			SetCol(0, q * Vec3(1.0f, 0.0f, 0.0f));
			SetCol(1, q * Vec3(0.0f, 1.0f, 0.0f));
			SetCol(2, q * Vec3(0.0f, 0.0f, 1.0f));
		}

		CUDA_CALLABLE inline Vec3 GetCol(int index) const { return Vec3(m[index][0], m[index][1], m[index][2]); }
		CUDA_CALLABLE inline void SetCol(int index, const Vec3 &v)
		{
			m[index][0] = v.x;
			m[index][1] = v.y;
			m[index][2] = v.z;
		}

		Real m[3][3];
	};

	CUDA_CALLABLE inline Mat33 Outer(const Vec3 &a, const Vec3 &b)
	{
		return Mat33(a * b.x, a * b.y, a * b.z);
	}

	CUDA_CALLABLE inline Vec3 Multiply(const Mat33 &a, const Vec3 &v)
	{
		Vec3 result;
		::Mirage::MatrixMultiply<3, 3, 1>(&result.x, &a.m[0][0], &v.x);
		return result;
	}

	CUDA_CALLABLE inline Mat33 Multiply(const Mat33 &a, const Mat33 &b)
	{
		Mat33 result;
		::Mirage::MatrixMultiply<3, 3, 3>(&result.m[0][0], &a.m[0][0], &b.m[0][0]);
		return result;
	}

	CUDA_CALLABLE inline Mat33 Transpose(const Mat33 &a)
	{
		Mat33 t;
		::Mirage::MatrixTranspose<3, 3>(&t.m[0][0], &a.m[0][0]);
		return t;
	}

	CUDA_CALLABLE inline Real Determinant(const Mat33 &a)
	{
		return Dot(a.GetCol(0), Cross(a.GetCol(1), a.GetCol(2)));
	}

	CUDA_CALLABLE inline Vec3 operator*(const Mat33 &a, const Vec3 &v) { return Multiply(a, v); }
	CUDA_CALLABLE inline Mat33 operator*(const Real s, const Mat33 &a)
	{
		Mat33 result;
		::Mirage::MatrixScale<3, 3>(&result.m[0][0], &a.m[0][0], s);
		return result;
	}

	CUDA_CALLABLE inline Mat33 operator*(const Mat33 &a, const Real s) { return s * a; }
	CUDA_CALLABLE inline Mat33 operator*(const Mat33 &a, const Mat33 &b) { return Multiply(a, b); }

	CUDA_CALLABLE inline Mat33 operator-(const Mat33 &a, const Mat33 &b)
	{
		Mat33 result;
		::Mirage::MatrixSub<3, 3>(&result.m[0][0], &a.m[0][0], &b.m[0][0]);
		return result;
	}

	CUDA_CALLABLE inline Mat33 operator+(const Mat33 &a, const Mat33 &b)
	{
		Mat33 result;
		::Mirage::MatrixAdd<3, 3>(&result.m[0][0], &a.m[0][0], &b.m[0][0]);
		return result;
	}

	CUDA_CALLABLE inline Mat33 &operator+=(Mat33 &a, const Mat33 &b) { return a = a + b; }
	CUDA_CALLABLE inline Mat33 &operator-=(Mat33 &a, const Mat33 &b) { return a = a - b; }

	CUDA_CALLABLE inline Vec2 TransformVector(const Mat33 &a, const Vec2 &v)
	{
		Vec2 result;
		result.x = a.m[0][0] * v.x + a.m[1][0] * v.y;
		result.y = a.m[0][1] * v.x + a.m[1][1] * v.y;
		return result;
	}

	CUDA_CALLABLE inline Vec2 TransformPoint(const Mat33 &a, const Vec2 &v)
	{
		Vec2 result;
		result.x = a.m[0][0] * v.x + a.m[1][0] * v.y + a.m[2][0];
		result.y = a.m[0][1] * v.x + a.m[1][1] * v.y + a.m[2][1];
		return result;
	}

	// returns the skew-symmetric matrix that performs cross(v, x) when multiplied on the left
	CUDA_CALLABLE inline Mat33 Skew(const Vec3 &v)
	{
		return Mat33(0.0f, -v.z, v.y,
					 v.z, 0.0f, -v.x,
					 -v.y, v.x, 0.0f);
	}

} // namespace Mirage

CUDA_CALLABLE inline Mirage::Quat::Quat(const Mat33 &m)
{
	if (m.GetCol(2).z < 0)
	{
		if (m.GetCol(0).x > m.GetCol(1).y)
		{
			float t = 1 + m.GetCol(0).x - m.GetCol(1).y - m.GetCol(2).z;
			x = t * (0.5f / ::Mirage::Sqrt(t));
			y = (m.GetCol(0).y + m.GetCol(1).x) * (0.5f / ::Mirage::Sqrt(t));
			z = (m.GetCol(2).x + m.GetCol(0).z) * (0.5f / ::Mirage::Sqrt(t));
			w = (m.GetCol(1).z - m.GetCol(2).y) * (0.5f / ::Mirage::Sqrt(t));
		}
		else
		{
			float t = 1 - m.GetCol(0).x + m.GetCol(1).y - m.GetCol(2).z;
			x = (m.GetCol(0).y + m.GetCol(1).x) * (0.5f / ::Mirage::Sqrt(t));
			y = t * (0.5f / ::Mirage::Sqrt(t));
			z = (m.GetCol(1).z + m.GetCol(2).y) * (0.5f / ::Mirage::Sqrt(t));
			w = (m.GetCol(2).x - m.GetCol(0).z) * (0.5f / ::Mirage::Sqrt(t));
		}
	}
	else
	{
		if (m.GetCol(0).x < -m.GetCol(1).y)
		{
			float t = 1 - m.GetCol(0).x - m.GetCol(1).y + m.GetCol(2).z;
			x = (m.GetCol(2).x + m.GetCol(0).z) * (0.5f / ::Mirage::Sqrt(t));
			y = (m.GetCol(1).z + m.GetCol(2).y) * (0.5f / ::Mirage::Sqrt(t));
			z = t * (0.5f / ::Mirage::Sqrt(t));
			w = (m.GetCol(0).y - m.GetCol(1).x) * (0.5f / ::Mirage::Sqrt(t));
		}
		else
		{
			float t = 1 + m.GetCol(0).x + m.GetCol(1).y + m.GetCol(2).z;
			x = (m.GetCol(1).z - m.GetCol(2).y) * (0.5f / ::Mirage::Sqrt(t));
			y = (m.GetCol(2).x - m.GetCol(0).z) * (0.5f / ::Mirage::Sqrt(t));
			z = (m.GetCol(0).y - m.GetCol(1).x) * (0.5f / ::Mirage::Sqrt(t));
			w = t * (0.5f / ::Mirage::Sqrt(t));
		}
	}
}