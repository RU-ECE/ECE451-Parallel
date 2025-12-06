/*
 * Integrates conservative scalar equations
 * using a centred scheme, as originally provided in POM.
 *
 * Model grid is imax * jmax * kbot = 210 * 102 * 21.
 *
 * MPI: make the grid bigger so it is more accurate
 * 2100 * 1020 * 210 * time; a factor of 1000 more computation with time 10k.
 */

// Global sizes (including halo cells, like the Fortran code)
constexpr int IMAX = 210 + 2; // corresponds to Fortran im
constexpr int JMAX = 102 + 2; // corresponds to Fortran jm
constexpr auto KBOT = 21; // corresponds to Fortran kb
constexpr int KBM1 = KBOT - 1; // kb-1

// Convenience aliases for array types
using Array3 = double[IMAX][JMAX][KBOT];
using Array3Vel = double[IMAX][JMAX][KBOT]; // for u, v, w, aam
using Array3Flux = double[IMAX][JMAX][KBOT];
using Array2 = double[IMAX][JMAX];
using Array1 = double[KBOT];

// This is a direct translation of the Fortran advt1 subroutine.
// All loops are converted from 1..N (Fortran) to 0..N-1 (C++).
// Fortran indexes (i,j,k) map to C++ [i][j][k] with i,j,k zero-based.
void advection_temperature(Array3& fb, Array3& f, const Array3& fclim, Array3& ff, Array3Flux& xflux, Array3Flux& yflux,
						   Array3Flux& zflux, const Array2& dt, const Array3Vel& u, const Array3Vel& v,
						   const Array3Vel& w, const Array3Vel& aam, const Array2& h, const Array2& dum,
						   const Array2& dvm, const Array2& dx, const Array2& dy, const Array1& dz, const Array2& etb,
						   const Array2& etf, const Array2& art, const double tprni, const double dti2) {
	constexpr int im = IMAX;
	constexpr int jm = JMAX;
	constexpr int kb = KBOT;
	constexpr int kbm1 = KBM1;
	constexpr int imm1 = im - 1; // Fortran imm1 = im-1
	constexpr int jmm1 = jm - 1; // Fortran jmm1 = jm-1
	// ------------------------------------------------------------------
	// 1) Top boundary: f(:,:,kb) = f(:,:,kbm1), fb(:,:,kb) = fb(:,:,kbm1)
	//    Fortran:
	//      do j=1,jm
	//        do i=1,im
	//          f(i,j,kb)=f(i,j,kbm1)
	//          fb(i,j,kb)=fb(i,j,kbm1)
	//        end do
	//      end do
	// ------------------------------------------------------------------
	for (auto j = 0; j < jm; ++j) {
		for (auto i = 0; i < im; ++i) {
			f[i][j][kb - 1] = f[i][j][kbm1 - 1]; // kb → index kb-1; kbm1 → kb-2
			fb[i][j][kb - 1] = fb[i][j][kbm1 - 1];
		}
	}
	// ------------------------------------------------------------------
	// 2) Advective fluxes (horizontal)
	//
	// Fortran:
	//   do k=1,kbm1
	//     do j=2,jm
	//       do i=2,im
	//         xflux(i,j,k)=.25*((dt(i,j)+dt(i-1,j))
	//      $                  *(f(i,j,k)+f(i-1,j,k))*u(i,j,k))
	//         yflux(i,j,k)=.25*((dt(i,j)+dt(i,j-1))
	//      $                  *(f(i,j,k)+f(i,j-1,k))*v(i,j,k))
	// ------------------------------------------------------------------
	for (auto k = 0; k < kbm1; ++k) { // k: Fortran 1..kbm1 → 0..kbm1-1
		for (auto j = 1; j < jm; ++j) { // j: 2..jm → 1..jm-1
			for (auto i = 1; i < im; ++i) { // i: 2..im → 1..im-1
				xflux[i][j][k] = 0.25 * (dt[i][j] + dt[i - 1][j]) * (f[i][j][k] + f[i - 1][j][k]) * u[i][j][k];
				yflux[i][j][k] = 0.25 * (dt[i][j] + dt[i][j - 1]) * (f[i][j][k] + f[i][j - 1][k]) * v[i][j][k];
			}
		}
	}
	// ------------------------------------------------------------------
	// 3) Subtract fclim from fb
	//
	// Fortran:
	//   do k=1,kb
	//     do j=1,jm
	//       do i=1,im
	//         fb(i,j,k)=fb(i,j,k)-fclim(i,j,k)
	// ------------------------------------------------------------------
	for (auto k = 0; k < kb; ++k) {
		for (auto j = 0; j < jm; ++j)
			for (auto i = 0; i < im; ++i)
				fb[i][j][k] -= fclim[i][j][k];
	}
	// ------------------------------------------------------------------
	// 4) Add diffusive fluxes and scale by dx, dy
	//
	// Fortran:
	//   do k=1,kbm1
	//     do j=2,jm
	//       do i=2,im
	//         xflux(i,j,k)=xflux(i,j,k)
	//  $                    -.5*(aam(i,j,k)+aam(i-1,j,k))
	//  $                         *(h(i,j)+h(i-1,j))*tprni
	//  $                         *(fb(i,j,k)-fb(i-1,j,k))*dum(i,j)
	//  $                         /(dx(i,j)+dx(i-1,j))
	//         yflux(i,j,k)=yflux(i,j,k)
	//  $                    -.5*(aam(i,j,k)+aam(i,j-1,k))
	//  $                         *(h(i,j)+h(i,j-1))*tprni
	//  $                         *(fb(i,j,k)-fb(i,j-1,k))*dvm(i,j)
	//  $                         /(dy(i,j)+dy(i,j-1))
	//         xflux(i,j,k)=.5*(dy(i,j)+dy(i-1,j))*xflux(i,j,k)
	//         yflux(i,j,k)=.5*(dx(i,j)+dx(i,j-1))*yflux(i,j,k)
	// ------------------------------------------------------------------
	for (auto k = 0; k < kbm1; ++k) {
		for (auto j = 1; j < jm; ++j) {
			for (auto i = 1; i < im; ++i) {
				// Horizontal diffusion in X
				const auto diff_x = 0.5 * (aam[i][j][k] + aam[i - 1][j][k]) * (h[i][j] + h[i - 1][j]) * tprni *
					(fb[i][j][k] - fb[i - 1][j][k]) * dum[i][j] / (dx[i][j] + dx[i - 1][j]);
				// Horizontal diffusion in Y
				const auto diff_y = 0.5 * (aam[i][j][k] + aam[i][j - 1][k]) * (h[i][j] + h[i][j - 1]) * tprni *
					(fb[i][j][k] - fb[i][j - 1][k]) * dvm[i][j] / (dy[i][j] + dy[i][j - 1]);
				xflux[i][j][k] = xflux[i][j][k] - diff_x;
				yflux[i][j][k] = yflux[i][j][k] - diff_y;
				xflux[i][j][k] *= 0.5 * (dy[i][j] + dy[i - 1][j]);
				yflux[i][j][k] *= 0.5 * (dx[i][j] + dx[i][j - 1]);
			}
		}
	}
	// ------------------------------------------------------------------
	// 5) Add fclim back to fb
	//
	// Fortran:
	//   do k=1,kb
	//     do j=1,jm
	//       do i=1,im
	//         fb(i,j,k)=fb(i,j,k)+fclim(i,j,k)
	// ------------------------------------------------------------------
	for (auto k = 0; k < kb; ++k) {
		for (auto j = 0; j < jm; ++j)
			for (auto i = 0; i < im; ++i)
				fb[i][j][k] += fclim[i][j][k];
	}
	// ------------------------------------------------------------------
	// 6) Vertical advection (zflux)
	//
	// Fortran:
	//   do j=2,jmm1
	//     do i=2,imm1
	//       zflux(i,j,1)=f(i,j,1)*w(i,j,1)*art(i,j)
	//       zflux(i,j,kb)=0.
	//   do k=2,kbm1
	//     do j=2,jmm1
	//       do i=2,imm1
	//         zflux(i,j,k)=.5*(f(i,j,k-1)+f(i,j,k))*w(i,j,k)*art(i,j)
	// ------------------------------------------------------------------
	// k=1 and k=kb in Fortran -> k=0 and k=kb-1 in C++
	for (auto j = 1; j < jmm1; ++j) { // j: 2..jmm1 → 1..jmm1-1
		for (auto i = 1; i < imm1; ++i) { // i: 2..imm1 → 1..imm1-1
			zflux[i][j][0] = f[i][j][0] * w[i][j][0] * art[i][j];
			zflux[i][j][kb - 1] = 0.0;
		}
	}
	for (auto k = 1; k < kbm1; ++k) { // k: 2..kbm1 → 1..kbm1-1
		for (auto j = 1; j < jmm1; ++j)
			for (auto i = 1; i < imm1; ++i)
				zflux[i][j][k] = 0.5 * (f[i][j][k - 1] + f[i][j][k]) * w[i][j][k] * art[i][j];
	}
	// ------------------------------------------------------------------
	// 7) Add net fluxes and step forward in time
	//
	// Fortran:
	//   do k=1,kbm1
	//     do j=2,jmm1
	//       do i=2,imm1
	//         ff(i,j,k)=xflux(i+1,j,k)-xflux(i,j,k)
	//  $                 +yflux(i,j+1,k)-yflux(i,j,k)
	//  $                 +(zflux(i,j,k)-zflux(i,j,k+1))/dz(k)
	//
	//         ff(i,j,k)=(fb(i,j,k)*(h(i,j)+etb(i,j))*art(i,j)
	//  $                 -dti2*ff(i,j,k))
	//  $                 /((h(i,j)+etf(i,j))*art(i,j))
	// ------------------------------------------------------------------
	for (auto k = 0; k < kbm1; ++k) { // k: 1..kbm1 → 0..kbm1-1
		for (auto j = 1; j < jmm1; ++j) { // j: 2..jmm1 → 1..jmm1-1
			for (auto i = 1; i < imm1; ++i) { // i: 2..imm1 → 1..imm1-1
				ff[i][j][k] = (fb[i][j][k] * (h[i][j] + etb[i][j]) * art[i][j] -
							   dti2 *
								   (xflux[i + 1][j][k] - xflux[i][j][k] + yflux[i][j + 1][k] - yflux[i][j][k] +
									(zflux[i][j][k] - zflux[i][j][k + 1]) / dz[k])) /
					((h[i][j] + etf[i][j]) * art[i][j]);
			}
		}
	}
}
