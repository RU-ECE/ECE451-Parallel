/*
	Integrates conservative scalar equations
using a centred scheme, as originally provide in POM

 model grid is imax * jmax * kbot = 210 * 102 * 21


MPI, make the grid bigger so it's more accurate
 2100 * 1020 * 210 * time   a factor of 1000 more computation with time 10k
*/

enum { imax = 210 + 2, jmax = 102 + 2, kbot = 21 };
double fb[imax][jmax][kbot];
double f[imax][jmax][kbot];
double ff[imax][jmax][kbot];
//...


int rank; // this is my rank within MPI

/*
  gridsize = 4  a 4x4 grid of CPUs working on the problem
  row = rank / gridsize
  col = rank % gridsize

	0   1   2   3

  4   5   6   7

  8   9  10  11

 12  13  14  15


 double boundary_NORTHSOUTH[size];
 double boundary_EASTWEST[size];

copyboundary_to_matrix(f, EAST, boundary);


 */
void advection_temperature(fb,f,fclim,ff,xflux,yflux) {
//      real fb(im,jm,kb),f(im,jm,kb),fclim(im,jm,kb),ff(im,jm,kb)
//      real xflux(im,jm,kb),yflux(im,jm,kb)

	// first SEND my boundary data to my neighbors
	// then do some CPU while I wait for the answers to get back


	// you ALWAYS want these calls to be asynch for performance
	// but asynch is tougher to manage
  if (rank > gridsize) {
  	MPI_SEND(&f[1][1][0], imax-2, MPI_DOUBLE, rank-gridsize, ???, comm);
	}
	if (rank < gridsize*gridsize-gridsize) {
 	  MPI_SEND(&f[imax-2][1][0], imax-2, MPI_DOUBLE, rank+gridsize, ???, comm);
	}

	// ideally: SEND NORTH, SOUTH, EAST, WEST
	// RECV SOUTH, NORTH,WEST, EAST

	// DO WHATVER YOU CAN WITHOUT  THE DATA FROM YOUR NEIGHBORS
	// WAIT UNTIL COMPLETE

	// first do the internal boundary condition, giving my neighbors a chance to send me stucf
  for (auto j = 1; j < jmax-1; j++) {
		for (auto i = 1; i < imax-1; i++) {
			f[j][i][kbot-1] = f[j][i][kbot-2];
			f[j][i][kbot-1] = fb[j][i][kbot-2];
		}
	}

	// then wait for RECV

	MPI_RECV(..,)

	/*
		       0 1 2 3 ...   imax-2 imax-1
  0        B B B B ...   B      B
  1        B E E E ...   E      B
  2        B                    B
  3        B                    B




  jmax-2
  jmax-1  B B B B ...    B      B

	 */

	


	
	//     Do advective fluxes:
	
  for (auto k = 0; k < kbot-1; k++) {
		for (auto j = 1; j < jmax - 1; j++) {
			for (auto i = 1; i < imax - 1; i++) {
				xflux(i,j,k)=.25e0*((dt(i,j)+dt(i-1,j))
     $                          *(f(i,j,k)+f(i-1,j,k))*u(i,j,k))
            yflux(i,j,k)=.25e0*((dt(i,j)+dt(i,j-1))
     $                          *(f(i,j,k)+f(i,j-1,k))*v(i,j,k))
          end do
        end do
      end do
C
C     Add diffusive fluxes:
C
      do k=1,kb
        do j=1,jm
          do i=1,im
            fb(i,j,k)=fb(i,j,k)-fclim(i,j,k)
          end do
        end do
      end do
C
      do k=1,kbm1
        do j=2,jm
          do i=2,im
            xflux(i,j,k)=xflux(i,j,k)
     $                    -.5e0*(aam(i,j,k)+aam(i-1,j,k))
     $                         *(h(i,j)+h(i-1,j))*tprni
     $                         *(fb(i,j,k)-fb(i-1,j,k))*dum(i,j)
     $                         /(dx(i,j)+dx(i-1,j))
            yflux(i,j,k)=yflux(i,j,k)
     $                    -.5e0*(aam(i,j,k)+aam(i,j-1,k))
     $                         *(h(i,j)+h(i,j-1))*tprni
     $                         *(fb(i,j,k)-fb(i,j-1,k))*dvm(i,j)
     $                         /(dy(i,j)+dy(i,j-1))
            xflux(i,j,k)=.5e0*(dy(i,j)+dy(i-1,j))*xflux(i,j,k)
            yflux(i,j,k)=.5e0*(dx(i,j)+dx(i,j-1))*yflux(i,j,k)
          end do
        end do
      end do
C
      do k=1,kb
        do j=1,jm
          do i=1,im
            fb(i,j,k)=fb(i,j,k)+fclim(i,j,k)
          end do
        end do
      end do
C
C     Do vertical advection:
C
      do j=2,jmm1
        do i=2,imm1
          zflux(i,j,1)=f(i,j,1)*w(i,j,1)*art(i,j)
          zflux(i,j,kb)=0.e0
        end do
      end do
C
      do k=2,kbm1
        do j=2,jmm1
          do i=2,imm1
            zflux(i,j,k)=.5e0*(f(i,j,k-1)+f(i,j,k))*w(i,j,k)*art(i,j)
          end do
        end do
      end do
C
C     Add net horizontal fluxes and then step forward in time:
C
      do k=1,kbm1
        do j=2,jmm1
          do i=2,imm1
            ff(i,j,k)=xflux(i+1,j,k)-xflux(i,j,k)
     $                 +yflux(i,j+1,k)-yflux(i,j,k)
     $                 +(zflux(i,j,k)-zflux(i,j,k+1))/dz(k)
C
            ff(i,j,k)=(fb(i,j,k)*(h(i,j)+etb(i,j))*art(i,j)
     $                 -dti2*ff(i,j,k))
     $                 /((h(i,j)+etf(i,j))*art(i,j))
          end do
        end do
      end do
C
      return
C
      end
C
