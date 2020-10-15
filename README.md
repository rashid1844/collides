collision detection for vehicles with uncertainty in the position represented as a gaussian distribution
The implementation is based on an approximate but quick and accurate method.

import collide

### Functions:

* collide.collides(x1, y1, x2, y2, L1, W1, theta1, L2, W2, theta2)

* collide.plot(x1, y1, x2, y2, L1, W1, theta1, L2, W2, theta2)

* collide.collides(x1, y1, x2, y2, L1, W1, theta1, L2, W2, theta2, cov1, cov2, method, sample_size)







### Variables:

* x,y: position

* L,W: length and width

* theta: angle of the vehicle in radian


### In case of uncertinity:

* define cov1, cov2: 2x2 array

* method: 's':sampling with sample_size(has default value of 100), 'g' euclidean gaussian distribution, 'e' epsilon shadow

