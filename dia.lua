Dia = {}

function Dia:Create(n)

	local t = {}
	-- entry table
  	t._et = {}
	t._maxlen = n
	

  function t:maxlen()
	return self._maxlen
  end

  function t:curlen()
	return #self._et
  end

  function t:setlen(n)
	if n >= self._maxlen or n >= #self._et then
		self._maxlen = n
	else
		for i=1,self._maxlen - n do
			table.remove(self._et,1)
		end
		self._maxlen = n
	end
  end

  function t:push(v)
	if v then
		table.insert(self._et, v)	
		if table.getn(self._et) > self._maxlen then
			table.remove(self._et,1)
		end
	end
  end
  
  function t:dia()
  	return table.concat(self._et,"\n")
  end
  
  function t:clear()
  	self._et = {}
  end
  
  return t
end

