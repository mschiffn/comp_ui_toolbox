%
% superclass for all spatial discretization options
%
% author: Martin F. Schiffner
% date: 2019-02-20
% modified: 2019-08-01
%
classdef (Abstract) options_spatial

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_spatial( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure row vector for size
            if ~isrow( size )
                errorStruct.message = 'size must be a row vector!';
                errorStruct.identifier = 'options_spectral:NoRowVector';
                error( errorStruct );
            end

            % ensure nonempty positive integers
            mustBePositive( size );
            mustBeInteger( size );
            mustBeNonempty( size );

            %--------------------------------------------------------------
            % 2.) create spatial discretization options
            %--------------------------------------------------------------
            % repeat spatial discretization options
            objects = repmat( objects, size );

        end % function objects = options_spatial( size )

	end % methods

end % classdef (Abstract) options_spatial
