%
% superclass for all spatial discretizations
%
% author: Martin F. Schiffner
% date: 2019-02-17
% modified: 2019-05-16
%
classdef spatial

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        absorption_model ( 1, 1 ) absorption_models.absorption_model = absorption_models.time_causal( 0, 0.5, 1, 1540, 4e6, 1 ) % absorption model for the lossy homogeneous fluid
        str_name	% name of spatial discretization

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = spatial( absorption_models, strs_name )

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
            auxiliary.mustBeEqualSize( absorption_models, strs_name );

            %--------------------------------------------------------------
            % 2.) create spatial discretizations
            %--------------------------------------------------------------
            % repeat default spatial discretization
            objects = repmat( objects, size( absorption_models ) );

            % iterate spatial discretizations
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).absorption_model = absorption_models( index_object );
                objects( index_object ).str_name = strs_name{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = spatial( absorption_models, strs_name )

	end % methods

end % classdef spatial
