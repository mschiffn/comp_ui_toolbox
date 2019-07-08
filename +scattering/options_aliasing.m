%
% superclass for all anti-aliasing options
%
% author: Martin F. Schiffner
% date: 2019-04-09
% modified: 2019-07-08
%
classdef options_aliasing

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% enumeration
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	enumeration

        exclude, ...	% exclude coherent spatial aliasing from scattering operator
        include         % include coherent spatial aliasing in scattering operator

    end % enumeration

end % classdef options_aliasing
