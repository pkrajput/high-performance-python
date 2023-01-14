
import numpy as np
import pickle
import sys

from mpi4py import MPI

def spectrogram(t, y, window_positions_range, window_width = 4.0 * 2 * np.pi):
    
    t_size = len(t) // 2

    spectrogram = np.zeros((len(window_positions_range), t_size), dtype=np.float32)

    for i, window_position in enumerate(window_positions_range):
        window_function = np.exp(-(t - window_position) ** 2 / (2 * window_width**2))
        y_window = (y * window_function)

        values = np.abs(np.fft.fft(y_window)) ** 2
        spectrogram[i, :] = values[:t_size]

    return np.log(1 + spectrogram).reshape(-1)


comm = MPI.COMM_WORLD
start = MPI.Wtime()

rank = comm.Get_rank()
size = comm.Get_size()
root = 0

### read parameters of signal

with open(f't.pkl', 'rb') as f:
    t = pickle.load(f)

with open(f'y.pkl', 'rb') as f:
    y = pickle.load(f)

t_size = len(t) // 2


n_window_steps = 1000 if len(sys.argv) < 2 else int(sys.argv[1])
window_width = 4.0 * 2 * np.pi if len(sys.argv) < 3 else int(sys.argv[2])

window_positions = np.linspace(-30 * 2 * np.pi, 30 * 2 * np.pi, n_window_steps, dtype=np.float32)
frequencies = np.fft.fftfreq(len(y), d=(t[1] - t[0]) / (2 * np.pi))[:t_size]

window_positions_count = int(len(window_positions) / size)
window_positions_range = window_positions[rank * window_positions_count:(rank + 1) * window_positions_count]
spectrogram_complete = np.empty(t_size * len(window_positions), dtype=np.float32) if rank == root else None


spectrogram_range = spectrogram(t.astype(np.float32), y.astype(np.float32), window_positions_range, window_width)

comm.Gather(spectrogram_range, spectrogram_complete, root)
end = MPI.Wtime()

### creating dumps for parameters
if rank == root:

    with open(f'positions-{size}.pkl', 'wb') as f:
        pickle.dump(window_positions, f)
    
    with open(f'frequencies-{size}.pkl', 'wb') as f:
        pickle.dump(frequencies, f)

    with open(f'spectrogram-{size}.pkl', 'wb') as f:
        pickle.dump(spectrogram_complete.reshape(len(window_positions), t_size).T, f)
    
    with open(f'time-{size}.pkl', 'wb') as f:
        pickle.dump(end - start, f)
