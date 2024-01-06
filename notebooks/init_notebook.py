import os, sys
currentFolder = os.path.abspath('')
try:
    sys.path.remove(str(currentFolder))
except ValueError: # Already removed
    pass

projectFolder = r'/home/abjawad/Documents/GitHub/multi-region-attention'


sys.path.append(str(projectFolder))
os.chdir(projectFolder)
print( f"current working dir{os.getcwd()}")

huggingface_cli_token = "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"
