import enum


class StainType(enum.Enum):
    ssDNA = 'ssdna'.upper()
    DAPI = 'dapi'.upper()
    HE = 'HE'.upper()
    mIF = 'mIF'.upper()


class CellBinElement(object):
    def __init__(self):
        self.schedule = None
        self.task_name = ''
        self.sub_task_name = ''


if __name__ == '__main__':
    st = StainType
    print(st.ssDNA.name, st.ssDNA.value)