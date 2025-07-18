from pathlib import Path

import numpy as np

from cellbin2.image import cbimwrite
from cellbin2.modules.metadata import ProcFile
from cellbin2.modules import naming
from cellbin2.matrix.matrix import cMatrix
from cellbin2.utils.stereo_chip import StereoChip
from cellbin2.utils.config import Config


def extract4stitched(
        image_file: ProcFile,
        param_chip: StereoChip,
        m_naming: naming.DumpMatrixFileNaming,
        config: Config,
        detect_feature: bool = True,
):
    cm = cMatrix()
    cm.read(file_path=Path(image_file.file_path))
    cm.check_standards(config.genetic_standards)
    if detect_feature:
        cm.detect_feature(ref=param_chip.fov_template,
                          chip_size=min(param_chip.chip_specif))
        np.savetxt(m_naming.matrix_template, cm.template.template_points)
    cbimwrite(m_naming.heatmap, cm.heatmap)
    return cm


def extract4matrix(
        p_naming: naming.DumpPipelineFileNaming,
        image_file: ProcFile,
        m_naming: naming.DumpMatrixFileNaming,
):
    # 细胞mask提矩阵是否考虑组织mask，需确认下
    from cellbin2.matrix.matrix import save_cell_bin_data, save_tissue_bin_data
    cell_mask_path = p_naming.final_nuclear_mask
    tissue_mask_path = p_naming.final_tissue_mask
    cell_correct_mask_path = p_naming.final_cell_mask
    c_inp = None
    if Path(tissue_mask_path).exists():
        save_tissue_bin_data(
            image_file.file_path,
            str(m_naming.tissue_bin_matrix),
            tissue_mask_path,
        )
        c_inp = m_naming.tissue_bin_matrix
    if c_inp is None:
        c_inp = image_file.file_path
    if Path(cell_mask_path).exists():
        save_cell_bin_data(
            c_inp,
            str(m_naming.cell_bin_matrix),
            cell_mask_path)
    if Path(cell_correct_mask_path).exists():
        save_cell_bin_data(
            c_inp,
            str(m_naming.cell_correct_bin_matrix),
            cell_correct_mask_path
        )


def main():
    pass


if __name__ == '__main__':
    main()
