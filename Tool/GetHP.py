import win32gui
import win32api
import win32process
import ctypes
from ctypes import wintypes as w
import ctypes as c
import cv2
import time
from Tool.WindowsAPI import grab_screen

Psapi = ctypes.WinDLL('Psapi.dll')
Kernel32 = ctypes.WinDLL('kernel32.dll')
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010

def EnumProcessModulesEx(hProcess):
    buf_count = 256
    while True:
        LIST_MODULES_ALL = 0x03
        buf = (ctypes.wintypes.HMODULE * buf_count)()
        buf_size = ctypes.sizeof(buf)
        needed = ctypes.wintypes.DWORD()
        if not Psapi.EnumProcessModulesEx(hProcess, ctypes.byref(buf), buf_size, ctypes.byref(needed), LIST_MODULES_ALL):
            raise OSError('EnumProcessModulesEx failed')
        if buf_size < needed.value:
            buf_count = needed.value // (buf_size // buf_count)
            continue
        count = needed.value // (buf_size // buf_count)
        return map(ctypes.wintypes.HMODULE, buf[:count])

class Hp_getter():
    def __init__(self):
        hd = win32gui.FindWindow(None, "Hollow Knight")
        pid = win32process.GetWindowThreadProcessId(hd)[1]
        self.process_handle = win32api.OpenProcess(0x1F0FFF, False, pid)
        self.kernal32 = ctypes.windll.LoadLibrary(r"C:\\Windows\\System32\\kernel32.dll")

        print("process handle: ",int(self.process_handle))
        print(self.process_handle)
        
        self.hx = 0
        # get dll address
        hProcess = Kernel32.OpenProcess(
        PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
        False, pid)
        hModule  = EnumProcessModulesEx(hProcess)
        for i in hModule:
          temp = win32process.GetModuleFileNameEx(self.process_handle,i.value)
          if temp[-15:] == "UnityPlayer.dll":
            self.UnityPlayer = i.value
          if temp[-8:] == "mono.dll":
            self.mono = i.value
        
        # change the type of ReadProcessMemory
        self.kernal32.ReadProcessMemory.argtypes = w.HANDLE,w.LPCVOID,w.LPVOID,c.c_size_t,c.POINTER(c.c_size_t)
        self.kernal32.ReadProcessMemory.restype = w.BOOL
    
    def get_souls(self):
        base_address = self.mono + 0x001F72C0
        offset_address = ctypes.c_ulong()
        offset_list = [0x0, 0x54, 0x2D8, 0x18, 0x18, 0x120]
        self.kernal32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(offset_address), 4, None)
        for offset in offset_list:
          self.kernal32.ReadProcessMemory(int(self.process_handle), offset_address.value + offset, ctypes.byref(offset_address), 4, None)
        return offset_address.value

    def get_self_hp(self):
        base_address = self.mono + 0x001F50AC
        offset_address = ctypes.c_ulong()
        offset_list = [0x3B4, 0xC, 0x3C, 0x18, 0x10C, 0xE4]
        self.kernal32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(offset_address), 4, None)
        for offset in offset_list:
          self.kernal32.ReadProcessMemory(int(self.process_handle), offset_address.value + offset, ctypes.byref(offset_address), 4, None)
        return offset_address.value


    # This function can only get hp of hornet yet
    def get_boss_hp(self):
        # base_address = self.mono + 0x0020B504 
        base_address = self.UnityPlayer + 0x00FEF994 
        offset_address = ctypes.c_ulong()
        #offset_list = [0x10, 0x810, 0x0, 0x20, 0x10, 0xAC]
        offset_list = [0x54, 0x8, 0x1C, 0x1C, 0x7C, 0x18, 0xAC]
        self.kernal32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(offset_address), 4, None)
        for offset in offset_list:
          self.kernal32.ReadProcessMemory(int(self.process_handle), offset_address.value + offset, ctypes.byref(offset_address), 4, None)
        if offset_address.value > 900:
          return 901
        elif offset_address.value < 0:
          return -1
        return offset_address.value

    # the methods below can not work yet
    def get_play_location(self):
        x = ctypes.c_ulong()
        x.value += self.UnityPlayer + 0x00FEF994
        offset_list = [0x4C, 0x4, 0x4, 0x10, 0x0]
        self.kernal32.ReadProcessMemory(int(self.process_handle), x.value, ctypes.byref(x), 4, None)
        for offset in offset_list:
          self.kernal32.ReadProcessMemory(int(self.process_handle), x.value + offset, ctypes.byref(x), 4, None)
        xx = ctypes.c_float()
        self.kernal32.ReadProcessMemory(int(self.process_handle), x.value + 0x44, ctypes.byref(xx), 4, None)

        y = ctypes.c_ulong()
        y.value += self.UnityPlayer + 0x00FEF994
        offset_list = [0x24, 0x104, 0x6C, 0x10, 0xAC]
        self.kernal32.ReadProcessMemory(int(self.process_handle), y.value, ctypes.byref(y), 4, None)
        for offset in offset_list:
          self.kernal32.ReadProcessMemory(int(self.process_handle), y.value + offset, ctypes.byref(y), 4, None)

        yy = ctypes.c_float()
        self.kernal32.ReadProcessMemory(int(self.process_handle), y.value + 0xC, ctypes.byref(yy), 4, None)

        return xx.value, yy.value

    def get_hornet_location(self):
        # pic = cv2.cvtColor(grab_screen((0,0,1920,1080)), cv2.COLOR_RGBA2RGB)
        
        base_address = self.UnityPlayer + 0x00FEF994
        x = ctypes.c_ulong()
        offset_list = [0x20, 0x54, 0x24, 0x20, 0x5C]
        self.kernal32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(x), 4, None)
        for offset in offset_list:
          self.kernal32.ReadProcessMemory(int(self.process_handle), x.value + offset, ctypes.byref(x), 4, None)
          
        xx = ctypes.c_float()
        self.kernal32.ReadProcessMemory(int(self.process_handle), x.value + 0xC, ctypes.byref(xx), 4, None)

        base_address = self.UnityPlayer + 0x00FEF994
        y = ctypes.c_ulong()
        offset_list = [0x54, 0x8, 0x1C, 0x1C, 0x14]
        self.kernal32.ReadProcessMemory(int(self.process_handle), base_address, ctypes.byref(y), 4, None)
        for offset in offset_list:
          self.kernal32.ReadProcessMemory(int(self.process_handle), y.value + offset, ctypes.byref(y), 4, None)
     
        yy = ctypes.c_float()
        self.kernal32.ReadProcessMemory(int(self.process_handle), y.value + 0xAC, ctypes.byref(yy), 4, None)

        if xx.value > 14 and xx.value < 40:
          self.hx = xx.value
          
        # capture hornet's picture
        # self.get_hornet_pic(self.hx,yy.value,pic)
        
        return self.hx, yy.value
    
    # def get_hornet_pic(self,x,y,pic):
    #     x_center = int(-801.3+67.271*x )
    #     y_center = int(2750.12-65.779*y )
    #     # station_size = (x_center - 100, y_center - 200, x_center + 200, y_center + 200)
    #     #station = cv2.resize(cv2.cvtColor(grab_screen(station_size), cv2.COLOR_RGBA2RGB),(1000,500))
    #     #station = cv2.cvtColor(grab_screen(station_size), cv2.COLOR_RGBA2RGB)
    #     station = pic[y_center - 200:y_center+200, x_center - 200:x_center + 200]
    #     cv2.imwrite(f'../shots/{time.time()}.png',station)
    def get_hornet_pic(self):
        x, y = self.get_hornet_location()
        x_center = int(-801.3+67.271*x )
        y_center = int(2750.12-65.779*y )
        pic = cv2.cvtColor(grab_screen((0,0,1920,1080)), cv2.COLOR_RGBA2RGB)
        station = pic[y_center - 200:y_center+200, x_center - 200:x_center + 200]
        if len(station) != 0:
            return station
        
if __name__ == '__main__':
    from time import sleep
    from WindowsAPI import grab_screen
    sleep(2)
    hp = Hp_getter()
    print("player hp:",hp.get_self_hp())
    print("boss hp:",hp.get_boss_hp())
    print("souls:",hp.get_souls())
    print("player loc: ",hp.get_play_location())
    print("hornet loc: ",hp.get_hornet_location())
    
    # base_address = hp.mono + 0x001F50AC
    # offset_address = ctypes.c_ulong()
    # offset_list = [0x3B4, 0xC, 0x3C, 0x18, 0x10C, 0xE4]
    # hp.kernal32.ReadProcessMemory(int(hp.process_handle), base_address, ctypes.byref(offset_address), 4, None)
    # for offset in offset_list:
    #   hp.kernal32.ReadProcessMemory(int(hp.process_handle), offset_address.value + offset, ctypes.byref(offset_address), 4, None)
    # return offset_address.value
    
    # while True: 
    #     sleep(0.05)
    #     print("hornet loc: ",hp.get_hornet_location())
    
    