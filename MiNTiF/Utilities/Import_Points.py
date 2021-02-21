# (c) 2021, Luca Widmer  @  ETH Zurich
# Computer-assisted Applications in Medicine (CAiM) Group,  Prof. Orcun Goksel

from ij.gui import PointRoi
from ij.plugin.frame import RoiManager
#@ File (label='File', style='file', persist=true, required=true) filename
#@ImagePlus imp
"""
This script wil load points form a MINTIF coordinate.csv. and display them in the FIJI ROI Manager
"""

yourFile = open(str(filename), 'r')
RM = RoiManager()
rm = RM.getRoiManager()
# get current image scale
# some 2D Images always return (1,1,1)
calibration = imp.getCalibration()  # get the calibration, we need it for metadata
x_scale=round(calibration.pixelHeight, 3)
y_scale=round(calibration.pixelWidth,3)
z_scale=round(calibration.pixelDepth,3)

# write csv file
for i,line in enumerate(yourFile):
	try:
		# read header information
		if i ==1:
		# read  image extend
			Z_extend, X_extend, Y_extend= map(float, line.rstrip().split(','))

		elif i == 4:
			# read coordinate scale
			Z_scale,X_scale,Y_scale  = map(float, line.rstrip().split(','))
			Y_scale=round(Y_scale, 3)
			X_scale=round(X_scale, 3)
			Z_scale=round(Z_scale, 3)
			# if (X_scale, Y_scale, Z_scale) != (x_scale, y_scale, z_scale):
				# print((X_scale, Y_scale, Z_scale), "!=", (x_scale, y_scale, z_scale))
				# print("adapt coordinates to current image size")

		if i<6:
			continue
		# read coordinates
		#FIJI assumes inverted X,Y axis (X columns, Y rows)
		Z, X, Y= map( float, line.rstrip().split(',') )
		# print("original:\t",Z, X, Y)
		#rescale coordiantes to current image size and extend
		
		Y=(Y-Y_extend)/y_scale
		X=(X-X_extend)/x_scale
		Z=round((Z-Z_extend)/z_scale)
		# print("scaled:\t",Z, X, Y)
		roi = PointRoi(round(Y), round(X)) #  if you have rectangular ROI
		roi.setPosition(int(Z)) # set ROI Z-position/slice 
		rm.add(None, roi, int(Z)) # Trick to be able to set Z-position when less images than the number of ROI. line will appear as a digit index before the Roi Name  
	except:
		print("skip line {}".format(line))
# close file once done
yourFile.close()
				 
# Show All ROI + Associate ROI to slices  
rm.runCommand("Associate", "true")	 
# rm.runCommand("Show All with labels")