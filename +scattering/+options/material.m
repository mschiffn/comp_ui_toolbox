%
% enumerations for all material options
%
% author: Martin F. Schiffner
% date: 2019-04-09
% modified: 2019-08-03
%
classdef material

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% enumeration
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	enumeration

        compressibility, ...	% only compressibility fluctuations
        mass_density, ...       % only mass density fluctuations
        both                    % compressibility and mass density fluctuations

    end % enumeration

end % classdef material
