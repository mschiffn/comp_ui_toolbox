%
% enumeration for all spatial anti-aliasing filter states
%
% author: Martin F. Schiffner
% date: 2019-07-11
% modified: 2019-07-11
%
classdef options_anti_aliasing_status

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% enumeration
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	enumeration

        on, ...	% exclude coherent spatial aliasing from scattering operator (anti-aliasing)
        off     % include coherent spatial aliasing in scattering operator

    end % enumeration

end % classdef options_anti_aliasing_status
