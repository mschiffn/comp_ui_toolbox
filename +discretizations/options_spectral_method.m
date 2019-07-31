%
% enumeration for all types of spectral discretizations
%
% author: Martin F. Schiffner
% date: 2019-07-17
% modified: 2019-07-18
%
classdef options_spectral_method

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% enumeration
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	enumeration

        signal, ...         % individual frequency axes for each mixed signal
        setting, ...        % common frequency axis for each pulse-echo measurement
        sequence, ...       % common frequency axis for all pulse-echo measurements
        sequence_custom     % common frequency axis for all pulse-echo measurements using custom recording time interval

    end % enumeration

end % classdef options_spectral_method
