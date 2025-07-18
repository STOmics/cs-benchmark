import os.path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from cellbin2.utils import clog


class CellBinMatrix(object):
    """ 单个矩阵管理，是 cellbin2 生成的矩阵, 最终会以anndata的文件格式在内部操作，保存的时候，会转成gef的方式供后续读写 """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self._sn = None
        self._matrix_type = os.path.basename(self.file_path).split(".")[1]
        # class 类内部部访问数据
        self._stereo_exp = None  # raw stereo_exp
        self._cluster_exp = None

    @property
    def raw_data(self):
        if self._stereo_exp is None:
            self._stereo_exp = self.read(self.file_path)
            self._stereo_exp.cells["MID_counts"] = np.sum(self.raw_data.exp_matrix, axis=1)
        return self._stereo_exp

    @property
    def cluster_data(self):
        from stereo.io import read_stereo_h5ad
        # if self._cluster_exp==None:
        #     self._cluster_exp=read_stereo_h5ad("Z:\data\SS200000135TL_D1_cluster.h5ad")
        # return self._cluster_exp
        if self._cluster_exp is None:
            import copy
            self._cluster_exp=copy.deepcopy(self.raw_data)
            self._cluster_exp=self._cluster_exp.tl.filter_cells(
                min_counts=200,
                min_genes=10,
                max_genes=2500,
                pct_counts_mt=5,
            )
            self._cluster_exp.tl.raw_checkpoint()
            self._cluster_exp.tl.normalize_total()
            self._cluster_exp.tl.log1p()
            self._cluster_exp.tl.highly_variable_genes(
                min_mean=0.0125,
                max_mean=3,
                min_disp=0.5,
                n_top_genes=5000,
                res_key='highly_variable_genes'
            )
            self._cluster_exp.plt.highly_variable_genes(res_key='highly_variable_genes')
            self._cluster_exp.tl.scale()
            self._cluster_exp.tl.pca(
                use_highly_genes=True,
                n_pcs=30,
                res_key='pca'
            )
            self._cluster_exp.tl.neighbors(

                pca_res_key='pca',
                n_pcs=30,
                res_key='neighbors'
            )
            # compute spatial neighbors
            self._cluster_exp.tl.spatial_neighbors(
                neighbors_res_key='neighbors',
                res_key='spatial_neighbors'
            )
            self._cluster_exp.tl.umap(pca_res_key='pca', neighbors_res_key='neighbors', res_key='umap')
            self._cluster_exp.tl.leiden(neighbors_res_key='neighbors', res_key='leiden')
            self._cluster_exp.tl.find_marker_genes(
                cluster_res_key='leiden',
                method='t_test',
                use_highly_genes=False,
                use_raw=True
            )
        return self._cluster_exp

    @property
    def sn(self):
        if self._sn is None:
            self._sn = self.raw_data.sn
        return self._sn

    @property
    def cell_diameter(self):
        if "cell_diameter" not in self.raw_data.cells.obs.columns:
            if "area" not in self.raw_data.cells.obs.columns:
                raise Exception("No area result in .gef")
            else:
                self.raw_data.cells["cell_diameter"] = \
                    (2 * np.sqrt(self.raw_data.cells["area"].to_numpy() / np.pi) * self.raw_data.resolution / 1000)
        return self.raw_data.cells["cell_diameter"]

    @property
    def cell_n_gene(self):
        if "n_gene" not in self.raw_data.cells.obs.columns:
            self.raw_data.cells["n_gene"] = self.raw_data.exp_matrix.getnnz(axis=1)
        return self.raw_data.cells["n_gene"]

    @property
    def cell_MID_counts(self, ):
        if "MID_counts" not in self.raw_data.cells.obs.columns:
            self.raw_data.cells["MID_counts"] = np.sum(self.raw_data.exp_matrix,axis=1)
        return self.raw_data.cells["MID_counts"]

    def read(self,gef_path):
        if gef_path.endswith(".gem"):
            from stereo.io import read_gem

            data = read_gem(gef_path,bin_type='cell_bins',sep="\t",is_sparse=True)

            return data
        elif gef_path.endswith(".gef"):
            from stereo.io import read_gef, read_gef_info

            clog.info("the gef file is: ", read_gef_info(self.file_path))
            data = read_gef(gef_path,bin_type="cell_bins")

            return data
        elif gef_path.endswith(".h5ad"):
            try:
                from stereo.io import read_stereo_h5ad

                data = read_stereo_h5ad(gef_path,bin_type="cell_bins")
                return data
            except:
                from stereo.io import read_ann_h5ad
                data = read_ann_h5ad(gef_path,spatial_key="spatial",bin_type="cell_bins")
                clog.info("This is CellBin anndata file")
                return data

    def get_cellcount(self):
        return self.raw_data.n_cells

    def get_cellarea(self, ):
        return self.raw_data.cells["area"].mean().astype(np.int16),self.raw_data.cells["area"].median().astype(np.int16)

    def get_genetype(self, ):
        return self.cell_n_gene.to_numpy().mean().astype(np.int16),np.median(self.cell_n_gene.to_numpy()).astype(np.int16)

    def get_MID(self, ):
        return (self.cell_MID_counts.to_numpy().mean().astype(np.int16),
                np.median(self.cell_MID_counts.to_numpy()).astype(np.int16))

    def get_total_MID(self, ):
        return self.raw_data._exp_matrix.sum()

    def _plt_vio(self, data, title="", xlabel="", color="Blues", save_path=r"./"):
        # fig = plt.figure()
        fig, ax = plt.subplots()
        sns.violinplot(y=data, color=color, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("")
        fig.savefig(save_path)

    def plot_statistic_vio(self, save_path):
        MID_file = os.path.join(save_path, f"scatter_{self.sn}_{self._matrix_type}_MID.png")
        celldiameter_file = os.path.join(save_path, f"scatter_{self.sn}_{self._matrix_type}_celldiameter.png")
        genetype_file = os.path.join(save_path, f"scatter_{self.sn}_{self._matrix_type}_genetype.png")

        self._plt_vio(self.cell_diameter, title="Univariate distribution of Cell Diameter(\xb5m)",
                      xlabel="Cell Diameter",
                      save_path=celldiameter_file,
                      color=mcolors.TABLEAU_COLORS["tab:blue"])

        self._plt_vio(self.cell_n_gene,
                      title="Univariate distribution of Gene Type",
                      xlabel="Gene Type",
                      save_path=genetype_file,
                      color=mcolors.TABLEAU_COLORS["tab:orange"])

        self._plt_vio(self.cell_MID_counts,
                      title="Univariate distribution of MID counts",
                      xlabel="MID counts",
                      save_path=MID_file,
                      color=mcolors.TABLEAU_COLORS["tab:green"])

    @property
    def celldensity(self, radiu=200):  # 200 pixel = 100 um
        if "cell_density" not in self.raw_data.cells.obs.columns:
            from sklearn.neighbors import radius_neighbors_graph

            A = radius_neighbors_graph(self.raw_data.position, radiu, include_self=False, metric="euclidean")
            adj_matrix = A.toarray()
            self.raw_data.cells.obs["cell_density"] = adj_matrix.sum(axis=0)
        return self.raw_data.cells.obs["cell_density"]

    def _plot_distribution(self, df, key, xlabel, title):
        import seaborn as sns

        # return plot
        pass

    def write_h5ad(self, save_path: str):
        from stereo.io import write_h5ad

        outkey_record = {'cluster': ['leiden']}
        write_h5ad(
            self.cluster_data,
            use_raw=True,
            use_result=True,
            key_record=outkey_record,
            output=os.path.join(save_path,f"{self.sn}_cluster.h5ad"),
        )

    def write_gef(self,save_path: str):
        from stereo.io import write_mid_gef

        write_mid_gef(self.raw_data,save_path)


class TissueMatrix(object):
    def __init__(self, file_path: str):
        self._file_path = file_path
        self._stereo_exp=None  # raw stereo_exp

    @property
    def tissue_exp(self, ):
        if self._stereo_exp is None:
            self._stereo_exp = self.read()
        return self._stereo_exp

    def read(self, ):
        if self._file_path.endswith(".gef"):
            from stereo.io import read_gef, read_gef_info

            clog.info("the gef file is: ", read_gef_info(self._file_path))
            data = read_gef(self._file_path, bin_size=100)
        return data

    def get_total_MID(self, ):
        return self.tissue_exp._exp_matrix.sum()
