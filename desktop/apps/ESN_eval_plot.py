#ESN_train.pyで学習済みのModel esn_weights_reference6.npyを使用し,ESN_record_trajectory.pyで作成した reference_trajectory_6.csvのenc_deg列において,enc_deg[t]→enc_deg[t+1]を予想し,実際のenc_deg[t+1]との誤差を評価するコード
import sys
sys.path.insert(0, 'desktop/apps')
import numpy as np
import matplotlib.pyplot as plt
import ESN as esnmod
import ESN_train as trainer

CSV = 'data/recorded_trajectory/csv/reference_trajectory_6.csv'
WPATH = 'data/esn/esn_weights_reference6.npy'
OUTDIR = 'data/esn'
OUTPNG = OUTDIR + '/esn_following_reference6.png'

def main():
    q = trainer.load_qdes(CSV)
    U = q[:-1]
    D = q[1:]
    Wout = np.load(WPATH)

    esn = esnmod.ESN(U.shape[1], D.shape[1], 200, density=0.1, input_scale=0.7, rho=0.99, leaking_rate=0.7)
    esn.Output.setweight(Wout)

    Ypred = esn.predict(U)
    rmse = np.sqrt(((Ypred - D)**2).mean())
    print('RMSE:', rmse)

    # 時系列プロット
    t = np.arange(len(D))
    plt.figure(figsize=(8,4))
    if D.shape[1] == 1:
        plt.plot(t, D.flatten(), label='actual')
        plt.plot(t, Ypred.flatten(), label='pred')
    else:
        for i in range(D.shape[1]):
            plt.plot(t, D[:,i], label=f'actual_{i}')
            plt.plot(t, Ypred[:,i], '--', label=f'pred_{i}')
    plt.legend()
    plt.xlabel('step')
    plt.ylabel('angle / deg')
    plt.title(f'ESN 1-step following RMSE={rmse:.4f}')
    plt.tight_layout()
    import os
    os.makedirs(OUTDIR, exist_ok=True)
    plt.savefig(OUTPNG)
    print('Saved plot ->', OUTPNG)

if __name__=='__main__':
    main()
