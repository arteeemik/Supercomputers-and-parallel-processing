#include "fdtd-2d.h"

double bench_t_start, bench_t_end;

static
double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, NULL);
    if (stat != 0)
        printf("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start()
{
    bench_t_start = rtclock();
}

void bench_timer_stop()
{
    bench_t_end = rtclock();
}

void bench_timer_print()
{
    printf("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}

static
void init_array(int tmax,
                int nx,
                int ny,
                float ex[nx][ny],
                float ey[nx][ny],
                float hz[nx][ny],
                float _fict_[tmax])
{
    int i, j;
    for(i = 0; i < tmax; i++)
        _fict_[i] = (float) i;
    for(i = 0; i < nx; i++)
        for(j = 0; j < ny; j++) {
            ex[i][j] = ((float) i * (j + 1)) / nx;
            ey[i][j] = ((float) i * (j + 2)) / ny;
            hz[i][j] = ((float) i * (j + 3)) / nx;
        }
}

static
void print_array(int nx,
                 int ny,
                 float ex[nx][ny],
                 float ey[nx][ny],
                 float hz[nx][ny])
{
    int i, j;
    fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
    fprintf(stderr, "begin dump: %s", "ex");
    for(i = 0; i < nx; ++i)
        for(j = 0; j < ny; ++j) {
            if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
            fprintf(stderr, "%0.2f ", ex[i][j]);
        }
    fprintf(stderr, "\nend   dump: %s\n", "ex");
    fprintf(stderr, "==END   DUMP_ARRAYS==\n");

    fprintf(stderr, "begin dump: %s", "ey");
    for(i = 0; i < nx; ++i)
        for(j = 0; j < ny; ++j) {
            if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
            fprintf(stderr, "%0.2f ", ey[i][j]);
        }
    fprintf(stderr, "\nend   dump: %s\n", "ey");

    fprintf(stderr, "begin dump: %s", "hz");
    for(i = 0; i < nx; ++i)
        for(j = 0; j < ny; ++j) {
            if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
            fprintf(stderr, "%0.2f ", hz[i][j]);
        }
    fprintf(stderr, "\nend   dump: %s\n", "hz");
}

static
void kernel_fdtd_2d(int tmax,
                    int nx,
                    int ny,
                    float ex[nx][ny],
                    float ey[nx][ny],
                    float hz[nx][ny],
                    float _fict_[tmax])
{
    int t, i, j;
    for(t = 0; t < tmax; ++t) {
        #pragma omp parallel for private(j, i) shared(ey, nx, ny, hz)
        for(i = 1; i < nx; ++i) {
            for(j = 0; j < ny; ++j) {
                ey[i][j] -= 0.5f * (hz[i][j] - hz[i - 1][j]);
            }
        }
        #pragma omp parallel for private(j, i) shared(ex, nx, ny, hz)
        for(i = 0; i < nx; ++i) {
            for (j = 1; j < ny; ++j) {
                ex[i][j] -= 0.5f * (hz[i][j] - hz[i][j - 1]);
            }
        }

        #pragma omp parallel for private(j) shared(ex, ey, nx, ny, hz)
        for (int j = 0; j < ny - 1; ++j) {
            hz[0][j] -= 0.7f * (ex[0][j + 1] - ex[0][j] + ey[1][j] - _fict_[t]);
        }

        #pragma omp parallel for private(j, i) shared(ex, ey, nx, ny, hz)
        for(i = 1; i < nx - 1; ++i) {
            for (j = 0; j < ny - 1; ++j) {
                hz[i][j] -= 0.7f * (ex[i][j + 1] - ex[i][j] + ey[i + 1][j] - ey[i][j]);
            }
        }
    }
    for (j = 0; j < ny; j++)
        ey[0][j] = _fict_[tmax - 1];
}

int main(int argc, char **argv)
{
    int tmax = TMAX;
    int nx = NX;
    int ny = NY;
    float (*ex)[nx][ny];
    float (*ey)[nx][ny];
    float (*hz)[nx][ny];
    float (*_fict_)[tmax];

    ex = (float (*)[nx][ny]) malloc((nx) * (ny) * sizeof(float));

    ey = (float (*)[nx][ny]) malloc((nx) * (ny) * sizeof(float));

    hz = (float (*)[nx][ny]) malloc((nx) * (ny) * sizeof(float));

    _fict_ = (float (*)[tmax]) malloc((tmax) * sizeof(float));

    init_array(tmax, nx, ny,
               *ex,
               *ey,
               *hz,
               *_fict_);

    bench_timer_start();
    kernel_fdtd_2d(tmax, nx, ny,
                   *ex,
                   *ey,
                   *hz,
                   *_fict_);

    bench_timer_stop();
    bench_timer_print();

    //print_array(nx, ny, *ex, *ey, *hz);

    free((void *) ex);
    free((void *) ey);
    free((void *) hz);
    free((void *) _fict_);

    return 0;
}
