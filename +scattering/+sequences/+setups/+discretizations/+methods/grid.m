% abstract superclass for all spatial discretization methods using grids
%
% author: Martin F. Schiffner
% date: 2019-08-20
% modified: 2019-10-21
%
classdef (Abstract) grid < scattering.sequences.setups.discretizations.methods.method

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = grid( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create spatial discretization methods using grids
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.setups.discretizations.methods.method( size );

        end % function objects = grid( size )

	end % methods

end % classdef (Abstract) grid < scattering.sequences.setups.discretizations.methods.method
