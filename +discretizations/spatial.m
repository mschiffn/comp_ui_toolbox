%
% superclass for all spatial discretizations
%
% author: Martin F. Schiffner
% date: 2019-02-17
% modified: 2019-06-02
%
classdef spatial

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        homogeneous_fluid ( 1, 1 ) pulse_echo_measurements.homogeneous_fluid	% properties of the lossy homogeneous fluid
        str_name = 'default'                                                    % name spatial discretization

    end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spatial( homogeneous_fluids, strs_name )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % return empty object if no arguments
            if nargin == 0
                return;
            end

            % ensure cell array for strs_name
            if ~iscell( strs_name )
                strs_name = { strs_name };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( homogeneous_fluids, strs_name );

            %--------------------------------------------------------------
            % 2.) create spatial discretizations
            %--------------------------------------------------------------
            % repeat default spatial discretization
            objects = repmat( objects, size( homogeneous_fluids ) );

            % iterate spatial discretizations
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).homogeneous_fluid = homogeneous_fluids( index_object );
                objects( index_object ).str_name = strs_name{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = spatial( homogeneous_fluids, strs_name )

	end % methods

end % classdef spatial
