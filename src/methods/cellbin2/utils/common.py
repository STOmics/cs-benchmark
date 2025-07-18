from enum import Enum
from cellbin2.modules import naming


class TechType(Enum):
    ssDNA = 1
    DAPI = 2
    HE = 3
    IF = 4
    Transcriptomics = 5
    Protein = 6
    Null = 7
    UNKNOWN = 10
    # Metabolome = 7


KIT_VERSIONS = (
    'Stereo-seq T FF V1.2',
    'Stereo-seq T FF V1.3',
    'Stereo-CITE T FF V1.0',
    'Stereo-CITE T FF V1.1',
    'Stereo-seq N FFPE V1.0',
)

KIT_VERSIONS_R = tuple(i + " R" for i in KIT_VERSIONS)

bPlaceHolder = False
fPlaceHolder = -999.999
iPlaceHolder = -999
sPlaceHolder = '-'

FILES_TO_KEEP = (
    naming.DumpImageFileNaming.registration_image,
    naming.DumpImageFileNaming.tissue_mask,
    naming.DumpImageFileNaming.cell_mask,
    naming.DumpPipelineFileNaming.ipr,
    naming.DumpPipelineFileNaming.rpi,
    naming.DumpPipelineFileNaming.final_nuclear_mask,
    naming.DumpPipelineFileNaming.final_cell_mask,
    naming.DumpPipelineFileNaming.final_tissue_mask,
    naming.DumpPipelineFileNaming.input_json
)


class ErrorCode(Enum):
    init = 'value __doc__'
    qcFail = 1, 'image qc failed'
    missFile = 2, 'missing file'
    sizeInconsistent = 3, 'input images are not in the same size'
    weightDownloadFail = 4, 'weight file download fail'


def write_e2f():
    err_md = "../../docs/v2/error.md"
    markdown_content = "### Error codes and corresponding meanings\n\n| Error code | meaning |\n| :---: | :---: |\n"
    for error in list(ErrorCode):
        code = error.value
        comment = error.__doc__.strip() if error.__doc__ else ""
        markdown_line = f"| {code} | {comment} |\n"
        markdown_content += markdown_line

    print(markdown_content)
    with open(err_md, 'w') as file:
        file.write(markdown_content)


if __name__ == '__main__':
    write_e2f()
