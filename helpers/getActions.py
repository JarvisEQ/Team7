import numpy as np

# expect a numpy array of 132 
def getActions(Qs):
	
	actionArray = []
	units = []	
	
	# reshaping the array makes life easier
	Qs = np.reshape(Qs, (12,11))

	while len(actionArray) < 7 :
	
		# get the max Q
		action = np.unravel_index(np.argmax(Qs, axis=None), Qs.shape)
		
		# set action low so that we do not chose it again
		Qs[action] = float('-inf')
		
		# convert from a tuple to a np array
		action = np.array(action)
		action[0] += 1
		action[1] += 1		

		# check to see if the unit is already being moved
		if action[0] in units:
			continue	
		
		# add the unit to the unit chosen array
		units.append(action[0])		

		# append it to the action pair
		actionArray.append(action)
	
	# return the array
	return actionArray


