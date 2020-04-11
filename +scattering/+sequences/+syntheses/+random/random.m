%
% abstract superclass for all random incident waves
%
% author: Martin F. Schiffner
% date: 2020-04-08
% modified: 2020-04-08
%
classdef (Abstract) random < scattering.sequences.syntheses.wave

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = random( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create random incident waves
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.syntheses.wave( size );

        end % function objects = random( size )

	end % methods

end % classdef (Abstract) random < scattering.sequences.syntheses.wave
