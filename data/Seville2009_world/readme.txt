Basic guide for using the Ant World, Ant Routes and Ant Eye.




The first thing that should be loaded is the Ant World variables which are contained within world5000_gray.mat. To use this load the .mat file. 4 variables shall be loaded into the workspace:


X = x coordinates of polygons (in metres) 

Y = y coordinates of polygons (in metres) 
Z = z corodinates of polygons (in metres)
colp = colour of polygons

See "Example.m" for a function that plots a top-down view of the world.

The next thing to load in an ant route. All the ant routes are contained within antRoutes.mat'. Each route is saved individually in e.g. Ant1_Route1, Ant1_Route2 etc. Each file contains a 3-column matric where the columns indicate:

[x co-ordinate,  y co-ordinate, orientation]   

as all routes are homeward the first entry indicates the nest location [630,845], and the final entry the nest [510,100]. Co-ordinates are spaced 1cm apart and the orientation is in degrees using the matlab convention.  Matrices vary in row lengths and the routes can be of various lengths. NOTE! Currently the routes are saved in centimetres but the world is saved in metres, thus when loading the route data be sure to convert to metres.
"Example.m" shows how to plot a route overlaid on the top-down view of the world.



The visual input perceived by an ant can be simulated using the ImgGrabber.m function.	This function requires 10 variables:


x = x co-ordinate of the ant in metres
y = y co-ordinate of the ant in metres
z = height in metres (should set to 0.01)
th = the orienation of the ant in degrees using the matlab convention

X = polygons variable as described above
Y = polygons variable as described above
Z = polygons variable as described above
colp = polygons variable as described above
hfov = the horizontal field of view in degrees to be covered (set to 296 to approximate ant eye)

resolution = the resolution in degrees of an individual ommatidium


 (set to 4 to approximate ant eye)


"Example.m" plots the view from the first position of Ant 1 Route 1.