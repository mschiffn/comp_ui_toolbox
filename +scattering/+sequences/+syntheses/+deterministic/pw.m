%
% superclass for all steered plane waves (PWs)
%
% author: Martin F. Schiffner
% date: 2020-04-08
% modified: 2020-07-27
%
classdef pw < scattering.sequences.syntheses.deterministic.qpw

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = pw( e_theta )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty e_theta
            if nargin < 1 || isempty( e_theta )
                e_theta = math.unit_vector( [ 1, 0, 0 ] );
            end

            % superclass ensures class math.unit_vector for e_theta

            %--------------------------------------------------------------
            % 2.) create steered plane waves (PWs)
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.sequences.syntheses.deterministic.qpw( e_theta );

        end % function objects = pw( e_theta )

	end % methods

end % classdef pw < scattering.sequences.syntheses.deterministic.qpw
