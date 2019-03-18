%
% superclass for all spectral discretization options
%
% author: Martin F. Schiffner
% date: 2019-02-20
% modified: 2019-03-18
%
classdef options_spectral

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% enumeration
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	enumeration

        signal, ...     % individual frequency axes for each mixed signal
        setting, ...    % common frequency axis for each pulse-echo measurement
        sequence        % common frequency axis for all pulse-echo measurements

    end % enumeration

end % classdef options_spectral
