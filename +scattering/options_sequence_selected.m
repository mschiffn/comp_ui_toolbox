%
% superclass for all selected sequence options
%
% author: Martin F. Schiffner
% date: 2019-07-30
% modified: 2019-07-30
%
classdef options_sequence_selected < scattering.options_sequence

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        indices ( :, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 1 % indices of selected sequential pulse-echo measurements

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_sequence_selected( indices )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for indices
            if ~iscell( indices )
                indices = { indices };
            end

            %--------------------------------------------------------------
            % 2.) create selected sequence options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options_sequence( size( indices ) );

            % iterate selected sequence options
            for index_object = 1:numel( objects )

                % property validation functions ensure valid column vector for indices{ index_object }

                % set independent properties
                objects( index_object ).indices = indices{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = options_sequence_selected( indices )

	end % methods

end % classdef options_sequence_selected < scattering.options_sequence
