function cep = readcep32(cepfile)
% Return a matrix containing the MFCC in its columns. 
% The cepfile should contain 32bit (single-precision float point) 

fid = fopen(cepfile,'r');
num_inputs = fread(fid,1,'int32');   
num_outputs = fread(fid,1,'int32');  
num_pats = fread(fid,1,'uint32');    

cep = zeros(num_inputs, num_pats);
for i=1:num_pats,
  cep(:,i) = fread(fid,num_inputs,'single');
  output = fread(fid,1,'single');
end

fclose(fid);