%
% superclass for all selected sequence options
%
% author: Martin F. Schiffner
% date: 2019-07-30
% modified: 2020-01-18
%
classdef sequence_selected < scattering.options.sequence

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
        function objects = sequence_selected( indices )

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
            objects@scattering.options.sequence( size( indices ) );

            % iterate selected sequence options
            for index_object = 1:numel( objects )

                % property validation functions ensure valid column vector for indices{ index_object }

                % set independent properties
                objects( index_object ).indices = indices{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = sequence_selected( indices )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( sequences_selected )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.options.sequence_selected
            if ~isa( sequences_selected, 'scattering.options.sequence_selected' )
                errorStruct.message = 'sequences_selected must be scattering.options.sequence_selected!';
                errorStruct.identifier = 'string:NoOptionsSequenceSelected';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "selected"
            strs_out = repmat( "selected", size( sequences_selected ) );

            % iterate selected sequence options
            for index_objects = 1:numel( sequences_selected )

                strs_out( index_objects ) = sprintf( "selected (%s)", strjoin( string( sequences_selected( index_objects ).indices ), ', ' ) );

            end % for index_objects = 1:numel( sequences_selected )

        end % function strs_out = string( sequences_selected )

	end % methods

end % classdef sequence_selected < scattering.options.sequence
