% Purpose:
%   Convert .evl file produced by score_gplda_w.m to a format that is acceptable by 
%   eval_by_bosaris.m
% Input:
%   plda_evlfile    - .evl file produced by score_plda_w.m
%   ndxfile         - .ndx file contains the full path of speech files
% Output:
%   bosaris_evlfile - .evl file suitable for Bosaris toolkit
% Format of plda_evlfile:
%   100396,tabfsa_sre12.sph,b,1.6729176
% Format of bosaris_evlfile
%   110115,.../r141_2_1/sp12-01/data/mic_int/iaaakw-idrzps/iaaeox_sre12.sph,a,-101.6809729
% Format of ndxfile
%   110115,.../r141_2_1/sp12-01/data/mic_int/iaaakw-idrzps/iaaeox_sre12.sph,a
% Example usage:
%   evl2evl('evl/fw60/gplda60_male_cc1_1024c.evl', '../../ndx/male/core-core_8k_male_cc1.ndx','../../evl/fw60/sre12_gplda60_male_cc1_1024c.evl');
%
function evl2evl(plda_evlfile, ndxfile, bosaris_evlfile)

% Create a hashtable storing key-value pairs <modelID_testID_chan,score>
pldaHash = java.util.Hashtable;

% Read plda_evlfile and store contents in pldaHash
fid = fopen(plda_evlfile,'rt');
tline = fgetl(fid);
while ischar(tline)
    field = rsplit(',', tline);
    key = strcat(field{1},'_',field{2},'_',field{3});
    if pldaHash.containsKey(key) == 0,
        pldaHash.put(key, field{4});               % Store <modelID_testID_chan,score>
    end
    tline = fgetl(fid);
end
fclose(fid);

% Read ndxfile
fip = fopen(ndxfile,'rt');
fop = fopen(bosaris_evlfile,'wt');
tline = fgetl(fip);
while ischar(tline)
    field = rsplit(',', tline);
    evlID = field{1}; evlUtt = field{2}; evlChan = field{3};
    field = rsplit('/', evlUtt);
    key = strcat(evlID,'_',field{length(field)},'_',evlChan);
    if pldaHash.containsKey(key) == 1,
        fprintf(fop,'%s,%s,%s,%s\n',evlID,evlUtt,evlChan,pldaHash.get(key));
    end
    tline = fgetl(fip);
end
fclose(fop);
fclose(fip);