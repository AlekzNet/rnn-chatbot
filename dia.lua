Dia = {}

function Dia:Create(n)

	-- entry table
  	self._et = {}
	self._maxlen = n
	return self	
end

  function Dia:maxlen()
	return self._maxlen
  end

  function Dia:curlen()
	return #self._et
  end

  function Dia:setlen(n)
	if n >= self._maxlen or n >= #self._et then
		self._maxlen = n
	else
		for i=1,self._maxlen - n do
			table.remove(self._et,1)
		end
		self._maxlen = n
	end
  end

  function Dia:push(v)
	if v then
		table.insert(self._et, v)	
		if table.getn(self._et) > self._maxlen then
			table.remove(self._et,1)
		end
	end
  end
  
  function Dia:dia()
  	return table.concat(self._et,"\n")
  end
  
  function Dia:clear()
  	self._et = {}
  end
  
