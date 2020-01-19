%
% superclass for all full sequence options
%
% author: Martin F. Schiffner
% date: 2019-07-30
% modified: 2020-01-18
%
classdef sequence_full < scattering.options.sequence

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = sequence_full( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vectors for size
            % superclass ensures positive integers for size

            %--------------------------------------------------------------
            % 2.) create full sequence options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@scattering.options.sequence( size );

        end % function objects = sequence_full( varargin )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( sequences_full )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.options.sequence_full
            if ~isa( sequences_full, 'scattering.options.sequence_full' )
                errorStruct.message = 'sequences_full must be scattering.options.sequence_full!';
                errorStruct.identifier = 'string:NoOptionsSequenceFull';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "full"
            strs_out = repmat( "full", size( sequences_full ) );

        end % function strs_out = string( sequences_full )

	end % methods

end % classdef sequence_full < scattering.options.sequence
