%
% abstract superclass for all deterministic incident waves
%
% author: Martin F. Schiffner
% date: 2020-04-08
% modified: 2020-07-14
%
classdef (Abstract) deterministic < scattering.sequences.syntheses.wave

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = deterministic( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure one argument
            narginchk( 1, 1 );

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create deterministic incident waves
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.syntheses.wave( size );

        end % function objects = deterministic( size )

	end % methods

end % classdef (Abstract) deterministic < scattering.sequences.syntheses.wave
