function LINES = readline(FILE,NLINE,TYPE,varargin)
%READLINE   Reads specific lines from an ascii file.
%
%   SYNTAX:
%     LINES = readline(FILE);                   % Reads first line.
%     LINES = readline(FILE,NLINE);             % Reads specified lines.
%     LINES = readline(FILE,NLINE,TYPE);        % Specifies output type.
%     LINES = readline(FILE,NLINE,TYPE,...,P/V);% Optional TEXTREAD inputs.
%
%   INPUT:
%     FILE  - File name (string).
%     NLINE - Should be one of:
%               a) A SINGLE INTEGER indicating the line number to be readed
%                  from the beggining (if positive) or from the End-of-File
%                  (if negative).
%               b) A ROW VECTOR of integers specifying the lines to be
%                  readed (may have negative values). 
%               c) A TWO ROW MATRIX specifying ranges of lines to be
%                  readed. See NOTE below for details.
%             DEFAULT: 1 (reads the first line of FILE)
%     TYPE  - Specifies the output type. One of
%                 'cell' or FALSE  - Forces cell output (TEXTREAD default)
%               'string' or TRUE   - Forces string output with empty spaces
%                                    as padding. See NOTE below.
%             DEFAULT: 'string' if NLINE is single but 'cell' otherwise.
%     P/V   - Pairwise optional inputs for the TEXTREAD function, except
%             'delimiter' and 'whitespace'.
%             DEFAULT: none
%             
%   OUTPUT:
%     LINES - Readed lines. CAREFULL: may be string or cell type.
%
%   DESCRIPTION:
%     This function reads specified line(s) from an ascii file from the
%     beggining (positive NLINE) or the end of file (negatives NLINE) in a
%     very easy, fast and clean way.
%
%     Then, the user may get what he is looking for by searching through
%     the cell of string or the string matrix input (see the last example
%     here inlcuded).
%
%   NOTE:
%     * Optional inputs use its DEFAULT value when not given or [].
%     * If NLINE is empty, [], the program does not reads anything at all.
%     * Negative values in NLINE indicates line number from the
%       End-of-File, that is, -1 goes for the last line, -2 for the
%       penultimate line, and so on.
%     * The program differentiates between a row vector and a column vector
%       NLINE so the user be able to get ranges of lines with negative
%       NLINE elements and without carring about the length of file. For
%       example:
%         >> readline(FILE,[1 -1]')
%       reads the whole file, but
%         >> readline(FILE,[1 -1])
%       reads only the first and last lines (sometimes desirable), same as
%         >> readline(FILE,[-1:1])
%       But
%         >> readline(FILE,[1:-1])
%       do not reads anything at all because NLINE is empty.
%     * TYPE string inputs may be as short as a single char 's' or 'c'.
%     * If the file is too large, use the 'bufsize' optional argument (see
%     TEXTREAD for details).
%
%   EXAMPLE:
%     FILE = 'readline.m';
%     % Reads from this file the:
%     readline(FILE)                       % First line
%     readline(FILE,68)                    % This line
%     readline(FILE,[65:68 70:72],'s')     % Examples except this one
%     readline(FILE,[1 4; -4 -1]','s')     % First and last 4 lines
%     L = readline(FILE,[-1 1]')           % Whole file backwards!
%     flipud(a(strncmp(strtrim(a),'%',1))) % Get all commented lines from L
%     
%   SEE ALSO:
%     FGETL, TEXTREAD, TEXTSCAN, DLMREAD, LOAD
%     and
%     SAVEASCII by Carlos Vargas
%     at http://www.mathworks.com/matlabcentral/fileexchange
%
%
%   ---
%   MFILE:   readline.m
%   VERSION: 3.0 (Jun 08, 2009) (<a href="matlab:web('http://www.mathworks.com/matlabcentral/fileexchange/authors/11258')">download</a>) 
%   MATLAB:  7.7.0.471 (R2008b)
%   AUTHOR:  Carlos Adrian Vargas Aguilera (MEXICO)
%   CONTACT: nubeobscura@hotmail.com

%   REVISIONS:
%   1.0      Released. (May 22, 2008).
%   2.0      Great improvement by using TEXTREAD instead of FGETL as
%            sugested by Urs (us) Schwarz. New easy way for the lines
%            inputs. Now accepts ascending and descending (negative)
%            ranges. Do not accept file identifier input anymore. (Jun 10,
%            2008)
%   3.0      Rewritten code. Added try catch error for string convertion.
%            Do not uses default 'bufsize' TEXTREAD option. Do not accept
%            Inf values in NLINE anymore. Negative NLINES values now
%            indicated number of line from the End-of-File, instead of from
%            the last line. Forces first 3 inputs positions. New pairwise
%            optional inputs for TEXTSCAN. (Jun 08, 2009)

%   DISCLAIMER:
%   readline.m is provided "as is" without warranty of any kind, under the
%   revised BSD license.

%   Copyright (c) 2008-2009 Carlos Adrian Vargas Aguilera


% INPUTS CHECK-IN
% -------------------------------------------------------------------------

% Checks number of inputs:
if     nargin<1
 error('CVARGAS:readline:notEnoughInputs',...
  'At least 1 input is required.')
elseif nargout>1
 error('CVARGAS:readline:tooManyOutputs',...
  'At most 1 output is allowed.')
end

% Check NLINE:
if nargin<2
 NLINE = 1;     % Reads only the first line (default)
else
 if size(NLINE,1)>2
  error('CVARGAS:readline:incorrectNlineMatrix',...
   'NLINE matrix input must have only two rows.')
 end
 if any(~isfinite(NLINE))
  error('CVARGAS:readline:incorrectNlineType',...
   'NLINE must be only finite integers.')
 end
 NLINE = round(NLINE); % forces integer values.
end

% Check TYPE:
if nargin<3
 TYPE = [];   % depends on NLINE
elseif ~isempty(TYPE)
 if isnumeric(TYPE), TYPE = logical(TYPE); end
 s = 'string';
 c = 'cell';
 switch lower(TYPE)
  case {s(1:min(length(TYPE),6)),true}
   % 'string'
   TYPE = true;
  case {c(1:min(length(TYPE),4)),false,0}
   % 'cell'
   TYPE = false;
  otherwise
   error('CVARGAS:readline:incorrectType', ...
    'TYPE must be one of ''string'' or ''cell''.')
 end
end

% Check varargin:
nv = length(varargin);
c = 1;
while (c<=(nv-1))
 d = 'delimiter';
 w = 'whitespace';
 switch lower(varargin{c})
  case d(1:min(length(varargin{c}),9))
   varargin(c:c+1) = [];
   nv = nv-2;
   warning('CVARGAS:readline:ignoredDelimiterInput',...
    'Ignored ''delimiter'' optional input.')
  case w(1:min(length(varargin{c}),10))
   varargin(c:c+1) = []; 
   nv = nv-2;
   warning('CVARGAS:readline:ignoredWhitespaceInput',...
    'Ignored ''whitespace'' optional input.')
  otherwise
   c = c+2;
 end
end

% -------------------------------------------------------------------------
% MAIN
% -------------------------------------------------------------------------

% Gets NLINE size:
[m,n] = size(NLINE);

% Checks TYPE:
if isempty(TYPE)
 if (m*n==1)
  TYPE = true;
 else
  TYPE = false;
 end
end

% Checks NLINE:
NLINE = round(NLINE); % Forces integer input
if isempty(NLINE) || ((m*n==1) && (NLINE==0))
 % Finish if no lines will be readed:
 if ~TYPE
  LINES = {};
 else
  LINES = [];
 end
 return
end

% Gets the maximum line to be readed: 
ineg = (NLINE<0);
maxl = max(NLINE(:));
if any(ineg(:))  % Forced to read the whole text
 maxl = -1;
end 

% Reads up to the maximum line in cell array via TEXTREAD:
LINES = textread(FILE,'%s',maxl,... 
 'delimiter' ,'\n'  ,...
 'whitespace',''    ,...
 'bufsize'   ,80000 ,...
 varargin{:});

% % CODE using FGETL instead of TEXTREAD:
% %  Reads up to the maximum line in cell array via FGETL loop:
% if exist(FILE)~=2
%  error('Readline:IncorrectFileName','Not found such file.')
% end
% fid = fopen(FILE);
% if maxl>0
%  LINES = cell(maxl,1);
%  for k==1:maxl
%   LINES{k} = fgetl(fid);
%   if ~ischar(LINES{k})
%    LINES(k:maxl) = [];
%    break
%   end
%  end
% else
%  k = 1;
%  while 1
%   LINES{k} = fgetl(fid);
%   if ~ischar(LINES{k})
%    LINES(k) = [];
%    break
%   end
%   k = k+1;
%  end
% end
% fclose(fid);

% SELECTS SPECIFIED LINES
% Changes negative values:
nlines      = length(LINES);
NLINE(ineg) = nlines + NLINE(ineg) + 1; % Changed from v3.0
% Checks if specified more lines than they exist:
ip = (NLINE>nlines); % Lines beyond end of files!
in = (NLINE<1);      % Lines before first line!
% Deletes lines outside range:
ibad          = all(ip,1)|all(in,1);
NLINE(:,ibad) = [];
ip(:,ibad)    = [];
in(:,ibad)    = [];
n             = size(NLINE,2);
if isempty(NLINE)
 % All lines outside:
 if ~TYPE
  LINES = {};
 else
  LINES = [];
 end
 warning('CVARGAS:readline:ignoredNlineInput',...
   'All specified lines lie outside file range.')
 return
end
if m==2
 % Truncates ranges to file range:
 if any(ip(:)) || any(in(:))
  NLINE(ip) = nlines;
  NLINE(in) = 1;
  warning('CVARGAS:readline:TruncatedNlineValues',...
   'NLINE elements larger than file length were truncated.')
 end
 % Generates ranges:
 nlinetemp = cell(1,n);
 for k = 1:n
  if NLINE(1,k)<=NLINE(2,k)
   nlinetemp{k} = NLINE(1,k):NLINE(2,k); 
  else
   nlinetemp{k} = NLINE(1,k):-1:NLINE(2,k); 
  end
 end
 NLINE = cell2mat(nlinetemp);
 if isempty(NLINE)
  if ~TYPE
   LINES = {};
  else
   LINES = [];
  end
  return
 end
end

%  Selects the specified lines:
LINES = LINES(NLINE);

% OUTPUTS CHECK-OUT
% -------------------------------------------------------------------------

%  Changes to string matrix:
if TYPE
 try
	 LINES = char(LINES);
 catch
  warning('CVARGAS:readline:charConvertionError', ...
   'Output could not be transformed from cell to string. Maybe too large.')  
 end
end


% [EOF]   readline.m